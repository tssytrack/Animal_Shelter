#%% import packages

import pandas as pd
import category_encoders as ce
import numpy as np
from pandas_profiling import ProfileReport
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from biokit import corrplot
import scipy
import scipy.cluster.hierarchy as sch
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from sklearn import preprocessing

#%%
class Cleaning:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.summary = None
        self.vif = None
        self.target_freq = None

    def get_summary(self):
        uniques = self.data.nunique()
        dtypes = self.data.dtypes
        missing = self.data.isnull().sum()

        report = pd.DataFrame(uniques)
        report.columns = ["uniques"]
        report["dtypes"] = dtypes
        report["missing"] = missing
        report["missing_pct"] = report.missing / self.data.shape[0]

        self.summary = report

    def categorical(self):
        nunique = self.data.nunique()
        binary_list = nunique[nunique == 2].index.tolist()
        self.data[binary_list] = self.data[binary_list].astype("category")
        # binary_list = self.summary()[self.summary["uniques"] == 2].index.tolist()
        # self.data[binary_list] = self.data[binary_list].astype("category")

        dtypes = self.data.dtypes
        object_list = dtypes[dtypes == "object"].index.tolist()
        # object_list = self.summary()[self.summary()["dtypes"] == "object"].index.tolist()
        self.data[object_list] = self.data[object_list].astype("category")

    def one_hot(self, target):
        y = self.data[target]
        x = self.data.drop(target, axis = 1)
        nunique = x.nunique()
        binary_list = nunique[nunique == 2].index.tolist()

        dtypes = x.dtypes
        object_list = dtypes[dtypes == "object"].index.tolist()

        cat_list = binary_list + object_list
        cat_list = list(set(cat_list))
        x = pd.get_dummies(x, prefix_sep = "_", columns = cat_list, prefix = None)

        self.data = pd.concat([x, y], axis = 1)

        types = self.data.dtypes
        one_hot = types[types == "uint8"].index.tolist()
        self.level_freq = self.data[one_hot].sum(axis = 0)


    def imputation(self, threshold):
        self.get_summary()
        # vars that need imputation
        imput_list = self.summary[(self.summary["missing_pct"] < threshold) & (self.summary["missing_pct"] > 0)]
        imputing = self.data[imput_list.index]

        # vars that don't contain any missings
        no_missing_list = self.summary[self.summary["missing_pct"] == 0]
        no_missing = self.data[no_missing_list.index]

        # impute categorical variables
        imputing_cat = imputing.select_dtypes(exclude="number")
        cat_var = imputing_cat.columns
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        cat_imputted = pd.DataFrame(cat_imputer.fit_transform(imputing_cat))
        cat_imputted.columns = cat_var
        cat_imputted = cat_imputted.astype("category")

        # imputing numerical variables
        imputing_num = imputing.select_dtypes(include="number")
        num_var = imputing_num.columns.tolist()
        num_var_suffix = [x + "_indicator" for x in num_var]
        num_var = num_var + num_var_suffix
        num_imputer = SimpleImputer(strategy="median", add_indicator=True)
        num_imputted = pd.DataFrame(num_imputer.fit_transform(imputing_num))
        num_imputted.columns = num_var
        num_imputted[num_var_suffix] = num_imputted[num_var_suffix].astype("category")

        imputed_data = pd.concat([cat_imputted, num_imputted], axis=1, sort=False)
        imputed_data = pd.concat([imputed_data, no_missing], axis=1, sort=False)

        self.data = imputed_data
        self.get_summary()

    def missing_visualization(self):
        sns.heatmap(self.data.isnull(), cbar=False)

    def multicollinearity(self):
        # Calculating VIF
        nums = self.data._get_numeric_data()

        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif

        nums = nums[vif_list]

        # Cluster the correlation matrix
        Corr = nums.corr()
        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]
        ind_sorted = pd.DataFrame(ind_sorted)
        ind_sorted.columns = ["clusters", "number"]
        ind_sorted["var"] = columns
        freq = ind_sorted["clusters"].value_counts()
        ind_sorted = ind_sorted.merge(freq, how="left", left_on="clusters", right_index=True)
        ind_sorted_noone = ind_sorted[ind_sorted["clusters_y"] != 1]

        # conduct non-parametric ANOVA to decide which variables need to be dropped
        cluster_list = np.unique(ind_sorted_noone["clusters_x"].values)
        drop_list = []
        for i in cluster_list:
            vars = ind_sorted_noone[ind_sorted_noone["clusters_x"] == i]["var"]
            corr = Corr.loc[vars, vars]
            corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
            cluster_num = np.ones(corr.shape[0]) * i
            cluster_num = cluster_num.reshape(corr.shape[0], -1)
            corr = np.concatenate([corr, cluster_num], axis=1)
            corr = pd.DataFrame(corr)
            corr.columns = ["row", "columns", "corr", "clusters"]
            corr = corr[corr["corr"] != 1]
            if corr.shape[0] == 1:
                value = np.array(corr["corr"])
                if value < 0.7:
                    continue
            uniques = np.unique(corr[["row", "columns"]].values)
            p_value = []
            for ii in uniques:
                x = self.data[self.data["TARGET"] == 1][ii]
                y = self.data[self.data["TARGET"] == 0][ii]
                test = stats.kruskal(x, y)
                p_value.append(test[1])

            min = [i for i, j in enumerate(p_value) if j == max(p_value)]
            drop = np.delete(uniques, min)
            for var in drop:
                drop_list.append(var)

        self.data.drop(drop_list, axis = 1, inplace = True)

    def vif_corr_map(self):
        nums = self.data._get_numeric_data()
        vif = pd.DataFrame()
        vif["factor"] = [variance_inflation_factor(nums.values, i) for i in range(nums.shape[1])]
        vif["features"] = nums.columns
        vif_list = vif[vif["factor"] >= 5]["features"]
        self.vif = vif
        nums = nums[vif_list]
        Corr = nums.corr()

        d = sch.distance.pdist(Corr.values)
        L = sch.linkage(d, method="complete")
        ind = sch.fcluster(L, 0.5 * d.max(), "distance")
        ind = ind.reshape(len(ind), -1)
        ind = np.concatenate((ind, np.arange(ind.shape[0]).reshape(ind.shape[0], -1)), axis=1)
        ind_sorted = ind[ind[:, 0].argsort()]
        columns = [nums.columns.tolist()[i] for i in list(ind_sorted[:, 1])]

        nums = nums.reindex(columns, axis = 1)
        Corr = nums.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(Corr, cmap="RdYlBu")
        plt.xticks(range(len(Corr.columns)), Corr.columns, rotation=90)
        plt.yticks(range(len(Corr.columns)), Corr.columns)
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=0.8)

    def get_target_freq(self, target):
        self.target_freq = self.data[target].value_counts()

#%%

cleaning = Cleaning("/Users/dauku/Desktop/Git/DavidKu_IAA2020/AnimalShelter/shelter_animal_data.csv")
cleaning.get_summary()
report = cleaning.summary

colnames = cleaning.data.columns.tolist()
colnames[-3] = "Adopted"
cleaning.data.columns = colnames

cleaning.data = cleaning.data.drop(["PetID", "Adoption Date", "Name"], axis = 1)
cleaning.data["Breed"] = cleaning.data.loc[:, ["Primary Breed", "Secondary Breed"]].fillna("").sum(axis = 1)
cleaning.data["Breed"] = cleaning.data.loc[:, "Primary Breed"].fillna("Missing") + "_" + cleaning.data.loc[:, "Secondary Breed"].fillna("")
cleaning.data["Breed"] = cleaning.data["Breed"].str.replace(r"_$", "", regex = True).str.strip()

cleaning.data["Color"] = cleaning.data["Primary Color"].fillna("Missing") + "_" + cleaning.data["Secondary Color"].fillna("")
cleaning.data["Color"] = cleaning.data["Color"].str.replace(r"_$", "", regex = True).str.strip()

cleaning.data = cleaning.data.drop(["Primary Breed", "Secondary Breed", "Primary Color", "Secondary Color", "Tertiary Color", "Breed"], axis = 1)

# one-hot encoding
cleaning.one_hot("Adopted")

#%% modeling
data = cleaning.data
freq = cleaning.level_freq

# days to adoption has the problem of complete seperation problem. Drop it for now
data = data.drop(["Days to Adoption", "Zip Code"], axis = 1)

from sklearn.model_selection import train_test_split
x = data.drop("Adopted", axis = 1)
y = data["Adopted"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 55)


label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)
y = pd.get_dummies(y, prefix_sep = "_", prefix = None)
y = y[["YES", "NO"]].values.astype(float)

# logit = sm.GLM(y, x, family = sm.families.Binomial())
# Lasso_results = logit.fit_regularized(alpha = 100, L1_wt = 1)
alpha = np.linspace(0, 100, 101)
model = Logit(y_train, x_train)
params = []
for a in alpha:
    rslt = model.fit_regularized(alpha = a, disp = False)
    params.append(rslt.params)
params = np.asarray(params)

plt.figure(figsize = (10, 5))
plt.clf()
plt.axes([0.1, 0.1, 0.67, 0.8])
ag = []
for k in range()

model = Logit(y_train, x_train)
rslt1 = model.fit_regularized(alpha = 100, disp = False)
rslt1.summary()
prediction = rslt1.predict(exog = x_test)

x1 = x.values
d = sm.datasets.scotland.load(as_pandas = False)
d1 = d.exog
d2 = d.endog

