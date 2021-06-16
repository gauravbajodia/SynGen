#importing required libraries
import os
import random
from random import randint, choice
import pandas as pd

class generate_from_scratch:
    def __init__(self, seed = None):
        """
        Initiates the class and creates a Faker() object for later data generation by other methods
        seed: User can set a seed parameter to generate deterministic, non-random output
        """
        from faker import Faker
        
        self.faker = Faker
        self.fake = Faker('en_IN')
        self.seed = seed
        self.randnum = randint(1, 9)
        
        #initializing license-plate state list
        self.state_initials = self._initialize_state_initials()
        #initializing email domain list
        self.domain_list =  self._initialize_email_domains()
            
        
    def _initialize_state_initials(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + os.sep + "state_initials.txt"
        
        state_initials = []
        with open(path) as f:
            state_initials = [str(line).strip() for line in f.readlines()]
            
        return state_initials
     
        
    def _initialize_email_domains(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + os.sep + "email_domains.txt"
        
        email_domains = []
        with open(path) as f:
            email_domains = [str(line).strip() for line in f.readlines()]
            
        return email_domains
    
    
    def phone_num(self, seed = None):
        """
        Generates 10 digit Indian phone number in xxxxx-xxxxx format
        seed: Currently not used. Uses seed from the syngen class if chosen by user
        """
        random.seed(self.seed)
        
        phone_format = "{p1}-{p2}"
        
        p1 = str(randint(70000, 99999))
        p2 = str(randint(0, 99999)).rjust(5, "0")
        
        return phone_format.format(p1 = p1, p2 = p2)
    
    
    def license_plate(self, seed = None):
        """
        Generates vehicle license plate in the format MH-31-VS-9999
        seed: Currently not used. Uses seed from the syngen class if chosen by user
        """
        random.seed(self.seed)
        
        license_plate_format = "{p1}{p2}{p3}{p4}{p5}{p6}{p7}"    #try without placeholders for '-'
        state_in = choice(self.state_initials)
        
        p1 = str(state_in)
        p2 = "-"
        p3 = "".join([str(randint(0, 9)) for i in range(2)])
        p4 = "-"
        p5 = "".join([chr(randint(65, 90)) for _ in range(2)])
        p6 = "-"
        p7 = "".join([str(randint(0, 9)) for _ in range(4)])
        
        return license_plate_format.format(p1 = p1, p2 = p2, p3 = p3, p4 = p4, p5 = p5, p6 = p6, p7 = p7)
        
    
    def college_regno(self, seed = None):
        """
        Generates a 15-character college registration number in the format 2018ACSC1101085
        seed: Currently not used. Uses seed from the syngen class if chosen by user
        """
        random.seed(self.seed)
        
        reg_no_format = "{p1}{p2}{p3}"
        dept_list = ["ACSC", "ACIV", "AIFT", "AETX", "AETC"]
        dept = choice(dept_list)
        
        p1 = str(randint(2010, 2022))
        p2 = "".join([str(dept)])
        p3 = "".join([str(randint(0, 9)) for _ in range (7)]) 
        
        return reg_no_format.format(p1 =p1, p2 = p2, p3 = p3)
    
    def email(self, name, seed = None):
        """
        Generates realistic email from first and last name and a random domain address
        seed: Currently not used. Uses seed from the syngen class if chosen by user
        """
        random.seed(self.seed)
        
        name = str(name)
        f_name = name.split()[0]
        l_name = name.split()[-1]
        
        choice_int = choice(range(10))
        
        domain = choice(self.domain_list)
        
        name_formats = [
            "{first}{last}",
            "{first}.{last}",
            "{first}_{last}",
            "{last}.{first}",
            "{last}_{first}",
            "{f}{last}",
            "{first}.{l}",
            "{first}_{l}"
            ]
        name_fmt_choice = choice(name_formats)
        name_combo = name_fmt_choice.format(f = f_name[0], l = l_name[0], first = f_name, last = l_name)
        
        if choice_int < 7:
            email = name_combo + "@" + str(domain)
        else:
            email = name_combo + str(randint(1, 99)) + "@" + str(domain)
            
        return email
    
    def gen_data_series(self, num = 10, data_type = "name", seed = None):
        #description needs edit
        """
        Returns a pandas series object with the desired number of entries and data type
        Data types available:
        - 
        -
        -
        -
        -
        -
        """
        if type(data_type) != str:
            raise ValueError(
                "Data type must be of type str, found " + str(type(data_type))
                )
        try:
            num = int(num)
        except:
            raise ValueError(
                "Number of samples must be a positive integer, found " + num
                )
            
        if num <= 0:
            raise ValueError(
                "Number of samples must be a positive integer, found " + num
                )
        num = int(num)
        fake = self.fake
        self.faker.seed(self.seed)
        
        func_lookup = {
            #-----Address-----#
            "address" : fake.address,
            "building_number" : fake.building_number,
            "city" : fake.city,
            "country" : fake.country,
            "country_code" : fake.country_code,
            "postcode" : fake.postcode,
            "street_address" : fake.street_address,
            "state" : fake.state,
            
            #-----Automotive-----#
            "license_plate" : self.license_plate,
            
            #-----Barcode-----#
            "ean" : fake.ean,
            
            #-----Color-----#
            "color" : fake.color,
            "color_name" : fake.color_name,
            "rgb_color" : fake.rgb_color,
            
            #-----Company-----#
            "company" : fake.company,
            
            #-----Credit Card-----#
            
            
            #-----Date-Time-----#
            "century" : fake.century,
            "date" : fake.date,
            "date_between" : fake.date_between,        #between today and last 30 years
            "date_this_century" : fake.date_this_century,
            "date_this_decade" : fake.date_this_decade,
            "date_this_year" : fake.date_this_year,
            "date_this_month" : fake.date_this_month,
            "date_time" : fake.date_time,
            "day_of_month" : fake.day_of_month,
            "day_of_week" : fake.day_of_week,
            "future_date" : fake.future_date,        #between today and next 30 days
            "month" : fake.month,
            "month_name" : fake.month_name,
            "time" : fake.time,
            "year" : fake.year,
            
            #-----Geographic-----#
            "coordinate" : fake.coordinate,
            "latitude" : fake.latitude,
            "longitude" : fake.longitude,
            
            #-----Internet-----#
            "company_email" : fake.company_email,
            "ipv4" : fake.ipv4,
            "ipv4_private" : fake.ipv4_private,
            "ipv6" : fake.ipv6,
            "mac_address" : fake.mac_address,
            "port_number" : fake.port_number,
            "url" : fake.url,
            "email" : fake.email,
            
            #-----Book ISBN-----#
            "isbn10" : fake.isbn10,
            "isbn13" : fake.isbn13,
            
            #-----Job-----#
            "job" : fake.job,
            
            #-----Text-----#
            "paragraph" : fake.paragraph,
            "sentence" : fake.sentence,
            "text" : fake.text,
            "word" : fake.word,
            
            #-----Miscellaneous-----#
            "boolean" : fake.boolean,
            "json" : fake.json,
            "md5" : fake.md5,        #hexadecimal MD5 hash
            "password" : fake.password,
            
            #-----Person-----#
            "first_name" : fake.first_name,
            "first_name_female" : fake.first_name_female,
            "first_name_male" : fake.first_name_male,
            "last_name" : fake.last_name,
            "name" : fake.name,
            "name_female" : fake.name_female,
            "name_male" : fake.name_male,
            "aadhaar_id" : fake.aadhaar_id,
            "prefix" : fake.prefix,
            "language_name" : fake.language_name,
            
            "phone_num" : self.phone_num,
            #"email" : self.email,
            "college_regno" : self.college_regno
            }
        
        if data_type not in func_lookup:
            raise ValueError(
                "Data type must be one of " + str(list(func_lookup.keys()))
                )
        
        datagen_func = func_lookup[data_type]

        return pd.Series((datagen_func() for _ in range(num)))
    

    def _validate_args(self, num, fields):
        try:
            num = int(num)
        except:
            raise ValueError(
                "Number must be a positive integer, found " + num
                )
            
        if num <= 0:
            raise ValueError(
                "Number must be a positive integer, found " + num
                )
        
        num_cols = len(fields)
        if num_cols < 0:
            raise ValueError(
                "Please provide at least one type of data field to be generated"
                )
    def gen_dataframe(
            self,
            num = 10,
            fields = ["name"],
            real_email=True,
            phone_no = True,
            seed = None
            ):
        
        self._validate_args(num, fields)
        
        df = pd.DataFrame(data = self.gen_data_series(num, data_type = fields[0]), columns = [fields[0]])
        
        for col in fields[1:]:
            
            df[col] = self.gen_data_series(num, data_type=col)
            
            if ("email" in fields) and ("name" in fields) and real_email:
                df["email"] = df["name"].apply(self.email)
            
        return df
    
    def gen_csv(
            self, 
            df):
        df.to_csv("./generated_dataset.csv")
        return "./generated_dataset.csv"
        
    def gen_excel(
        self, 
        df
        ):
        df.to_excel("./generated_dataset.xlsx")
        return "./generated_dataset.xlsx"
    
    def gen_table(
            self,
            df,
            num = 10,
            fields = [],
            real_email=True,
            phone_no = True,
            seed = None,
            db_file = None, 
            table_name = None, 
            primarykey = None, 
            ):
        import sqlite3
        
        if not db_file:
            conn = sqlite3.connect("NewGenratedDB.db")
            c = conn.cursor()
        else:
            conn = sqlite3.connect(str(db_file))
            c = conn.cursor()
            
        if type(primarykey) != str and primarykey is not None:
            print("Primary key type not identified. Not generating any tabe.")
            return None
        
        #if primary key is None, designating the first field as primary key
        if not primarykey:
            table_cols = "(" + str(fields[0]) + " varchar PRIMARY KEY NOT NULL,"
            for col in fields[1:-1]:
                table_cols += str(col) + " varchar,"
            table_cols += str(fields[-1]) + " varchar" + ")"
            
        else:
            pk = str(primarykey)
            if pk not in fields:
                print("Desired primary key is not in the list of fields provided, cannot generate the table!")
                return None
            
            table_cols = "(" + str(fields[0]) + " varchar, "
            for col in fields[1:-1]:
                if col == pk:
                    table_cols += str(col) + " varchar PRIMARY KEY NOT NULL,"
                else:
                    table_cols += str(col) + " varchar, "
            table_cols += str(fields[-1]) + " varchar" + ")"
            
        if not table_name:
            table_name = "Table1"
        else:
            table_name = table_name
            
        str_drop_table = "DROP TABLE IF EXISTS " + str(table_name) + ";"
        c.execute(str_drop_table)
        str_create_table = (
            "CREATE TABLE IF NOT EXISTS " + str(table_name) + table_cols + ";"
        )
        c.execute(str_create_table)
        
        
        # Create a temporary df
        temp_df = self.gen_dataframe(
            num=num,
            fields=fields,
            real_email=real_email,
            phone_no=phone_no,
        )
        
        
        # Use the dataframe to insert into the table
        for i in range(num):
            str_insert = (
                "INSERT INTO "
                + table_name
                + " VALUES "
                + str(tuple(temp_df.iloc[i]))
                + ";"
            )
            c.execute(str_insert)

        # Commit the insertions and close the connection
        conn.commit()
        conn.close()
        
#  MOD 2
       
class generate_from_data:
    
    def __init__(self, df):
        
        self.df = df
        
    def learn_and_generate(self, pkey= None, num = 100):
        
        
        from sdv.tabular import GaussianCopula
        from sdv.evaluation import evaluate
        
        model = GaussianCopula(primary_key=pkey)
        model.fit(self.df)
        syn_data = model.sample(num)
        # syn_data.to_csv("./generated_synthetic_dataset.csv")

        # from sdv.evaluation import evaluate , evaluate (syn_data, df)
        # ,evaluate(syn_data, df)
        index= evaluate(syn_data, self.df)
       
        return syn_data,index
        

    
import numpy as np
import pandas as pd
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics

class compare_algo:

    def __init__(self, df ):
        
        self.df = df 



    def linear_reg(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        x = self.df.loc[:, : xlist]
        y = self.df.loc[:, ylist]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        
        regressor=LinearRegression()
        regressor.fit(x_train, y_train)
        y_pred=regressor.predict(x_test)
        rscore=metrics.r2_score(y_test, x_test)
        mae = metrics.mean_absolute_error(y_test, y_pred) #MAE
        mse = metrics.mean_squared_error(y_test,y_pred) #MSE
        rsme = np.sqrt(metrics.mean_squared_error(y_test,y_pred)) #RMSE
        
        plt.scatter(x, y, color = 'red', linewidth = 3)
        plt.plot(x_test, y_pred, color = 'green', linewidth = 3)
        plt.xlabel(xlist)
        plt.ylabel(ylist)
        plt.savefig("./static/images/regressions/Linear Reg.png", bbox_inches = 'tight')
        plt.clf()
        
        return rscore, mae, mse, rsme
    
    def svmr(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        '''
        x = x.astype(float)
        y = y.astype(float)
        '''
        sc = StandardScaler()
        x = sc.fit_transform(x)
        y = sc.fit_transform(y)
                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
                
        model = SVR(kernel = 'rbf')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = sc.inverse_transform(y_pred)
        rscore=metrics.r2_score(y_test, x_test)
        mae = metrics.mean_absolute_error(y_test, y_pred) #MAE
        mse = metrics.mean_squared_error(y_test,y_pred) #MSE
        rsme = np.sqrt(metrics.mean_squared_error(y_test,y_pred)) #RMSE
        
        
        x_grid = np.arange(min(x), max(x), 0.1)
        x_grid = x_grid.reshape((len(x_grid), 1))
        plt.scatter(x, y, color = 'red')
        plt.plot(x_grid, model.predict(x_grid), color = 'green')
        plt.xlabel(xlist)
        plt.ylabel(ylist)
        plt.savefig("./static/images/regressions/Support Vector Machine-R.png", bbox_inches = 'tight')
        plt.clf()
        
        return rscore, mae, mse, rsme
    
    #Decision Tree - Regression
    def DTRegression(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        
        model = DecisionTreeRegressor(criterion = 'mse')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rscore=metrics.r2_score(y_test, x_test)
        mae = metrics.mean_absolute_error(y_test, y_pred) #MAE
        mse = metrics.mean_squared_error(y_test,y_pred) #MSE
        rsme = np.sqrt(metrics.mean_squared_error(y_test,y_pred)) #RMSE
        
        
        x_grid = np.arange(min(x), max(x), 0.01)
        x_grid = x_grid.reshape((len(x_grid), 1))
        plt.scatter(x, y, color = 'red')
        plt.plot(x_grid, model.predict(x_grid), color = 'green')
        plt.xlabel(xlist)
        plt.ylabel(ylist)
        plt.savefig("./static/images/regressions/Decision Tree-R.png", bbox_inches = 'tight')
        plt.clf()
        
        return rscore, mae, mse, rsme
    
    #Random Forest - Regression
    def RFRegression(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        
        model = RandomForestRegressor(criterion = 'mse')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rscore=metrics.r2_score(y_test, x_test)
        mae = metrics.mean_absolute_error(y_test, y_pred) #MAE
        mse = metrics.mean_squared_error(y_test,y_pred) #MSE
        rsme = np.sqrt(metrics.mean_squared_error(y_test,y_pred)) #RMSE
        
        x_grid = np.arange(min(x), max(x), 0.01)
        x_grid = x_grid.reshape((len(x_grid), 1))
        plt.scatter(x, y, color = 'red')
        plt.plot(x_grid, model.predict(x_grid), color = 'green')
        plt.xlabel(xlist)
        plt.ylabel(ylist)
        plt.savefig("./static/images/regressions/Random Forest-R.png", bbox_inches = 'tight')
        plt.clf()
        
        return rscore, mae, mse, rsme
    
    #Logistic Regression
    def logistic_reg(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        
        x_train=x_train.astype(float)
        x_test=x_test.astype(float)        
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        log_model = LogisticRegression()
        log_model.fit(x_train, y_train)
        y_pred = log_model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot = True)
        self.plot_cf_matrix(cm, title="Logistic Regression")
        plt.savefig("./static/images/classifications/logistic Reg -C.png", bbox_inches = 'tight')
        plt.clf()
        #classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        #print(classification_report)
        
        return accuracy
    
    #Decision Tree - Classifier
    def DTClassifier(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        model = DecisionTreeClassifier(criterion = 'entropy')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot = True)
        self.plot_cf_matrix(cm, title="Decision Tree CLassifier")
        plt.savefig("./static/images/classifications/Decision Tree -C.png", bbox_inches = 'tight')
        plt.clf()
        #classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        #print(classification_report)
        
        return accuracy
    
    #Support Vector Machine - Classifer
    def svmc(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        #print("Accuracy=", accuracy)
        cm = metrics.confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot = True)
        self.plot_cf_matrix(cm, title="SVM Classifier")
        plt.savefig("./static/images/classifications/Support Vector Machine -C.png", bbox_inches = 'tight')
        plt.clf()
        #classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        #print(classification_report)
        
        return accuracy
    
    #K-Nearest Neighbour(KNN) - Classifier
    def knn(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        k_range = range(1, 40)
        score = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors = k)
            knn.fit(x_train, y_train)
            y_predict = knn.predict(x_test)
            score.append(metrics.accuracy_score(y_test, y_predict))
        optimal_k = score.index(max(score)) + 2             #+2 because(+1 for index and +1 for stability in graph)
        
        classifier = KNeighborsClassifier(n_neighbors = optimal_k)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot = True)
        self.plot_cf_matrix(cm, title="kNN Classifier")
        plt.annotate('Optimal k = ' + str(optimal_k) , (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.savefig("./static/images/classifications/K-Nearest Neighbor -C.png", bbox_inches = 'tight')
        plt.clf()
        #classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        #print(classification_report)
        
        return accuracy, optimal_k
    
    #Random Forest - Classifier    
    def RFClassifier(self, xlist, ylist):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        y = self.df[ylist].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        model = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_pred, y_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot = True)
        self.plot_cf_matrix(cm, title="Random Forest Classifier")
        plt.savefig("./static/images/classifications/Random Forest -C.png", bbox_inches = 'tight')
        plt.clf()
        #classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        #print(classification_report)
        
        return accuracy
    
    #DBSCAN Clustering
    def dbscan(self, xlist, min_samples):
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        x = x.astype(float)
        sc = StandardScaler()
        x = sc.fit_transform(x)
        model = DBSCAN(eps = 0.35, min_samples = min_samples)
        model.fit(x)
        y_db = model.fit_predict(x)
        y_labels = model.labels_
        sscore = silhouette_score(x, y_labels) * 100
        #viz
        u_labels = np.unique(y_db)
        
        for i in u_labels:
            plt.scatter(x[y_labels == i, 0], x[y_labels == i, 1], label = i)
        
        #plt.scatter(x[:,0], x[:,1],c=y_db)
        plt.legend()
        plt.title("DBSCAN Clustering")
        plt.savefig("./static/images/unsupervised/DBSCAN.png", bbox_inches = 'tight')
        plt.clf()
        
        return sscore   

    #K-Means Clustering
    def kmeans(self, xlist, n_clusters):
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        xlist = xlist.split(",")
        x = self.df[xlist].values
        model = KMeans(n_clusters = n_clusters)
        model.fit(x)
        y_labels = model.fit_predict(x)
        sscore = silhouette_score(x, y_labels) * 100
        
        u_labels = np.unique(y_labels)
        
        for i in u_labels:
            plt.scatter(x[y_labels == i, 0], x[y_labels == i, 1], label = i)
        plt.legend()
        plt.title("K-Means Clustering")
        plt.savefig("./static/images/unsupervised/K-Means.png", bbox_inches = 'tight')
        plt.clf()
        
        return sscore
    
    #Function to display and save confusion matrix
    def plot_cf_matrix(self, cf,
                          group_names=['True Neg', 'False Pos', 'False Neg', 'True Pos'],
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
        
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]
    
        if len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks
    
        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks
    
        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks
    
        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])
    
    
        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))
    
            #if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy, precision, recall, f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""
    
    
        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')
    
        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False
    
    
        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize = figsize)
        sns.heatmap(cf, annot = box_labels, fmt = "", cmap = cmap, cbar = cbar, xticklabels = categories, yticklabels = categories)
    
        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)
        
        if title:
            plt.title(title)
       
        
# SUPERVISED ALGORITHMS#############

    def supervisedAlgos(self, xlist, ylist):

        algoList=["regressions","RSME","Linear Reg","Support Vector Machine-R","Decision Tree-R","Random Forest-R"]
        
        linear = self.linear_reg(xlist, ylist)
        svmr = self.svmr(xlist, ylist)
        dtr = self.DTRegression(xlist, ylist)
        rfr = self.RFRegression(xlist, ylist)
        
        rscores = [linear[0], svmr[0], dtr[0], rfr[0]]
        maes = [linear[1], svmr[1], dtr[1], rfr[1]]
        rsmes = [linear[3], svmr[3], dtr[3], rfr[3]]

        
        plt.plot(algoList[2:], maes, color = 'r', marker = '<', markersize = 10, label = 'MAE', linestyle = '--', linewidth = 3)
        plt.plot(algoList[2:], rsmes, color = 'b', marker = 's', markersize = 10, label = 'RSME', linestyle = '-', linewidth = 3)
        plt.title("Error Comparison Metric Graph")
        plt.grid()
        plt.legend()
        plt.xticks(rotation = 60)
        plt.savefig("./static/images/regressions/comparison.png", bbox_inches = 'tight')
        plt.clf()
        

        return min(rsmes),algoList[rsmes.index(min(rsmes))+2],rsmes,algoList,len(algoList)

# Classification ##################

    def classification(self, xlist, ylist):
        
        algoList = ["classifications","Accuracy","logistic Reg -C","Decision Tree -C","Support Vector Machine -C","K-Nearest Neighbor -C","Random Forest -C"]
        
        logistic = self.logistic_reg(xlist, ylist)
        dtr = self.DTClassifier(xlist, ylist)
        svmc = self.svmc(xlist, ylist)
        knn = self.knn(xlist, ylist)
        rfc = self.RFClassifier(xlist, ylist)
        
        accuracies = [logistic, dtr, svmc, knn[0], rfc]
        
        plt.plot(algoList[2:], accuracies, color = 'b', marker = 'o', markersize = 10, label = "Accuracy", linestyle = '--', linewidth = 3)
        plt.title("Accuracy Comparison Graph")
        plt.grid()
        plt.legend()
        plt.xticks(rotation = 60)
        plt.savefig("./static/images/classifications/comparison.png", bbox_inches = 'tight')
        plt.clf()
        
        return max(accuracies),algoList[accuracies.index(max(accuracies))+2],accuracies,algoList,len(algoList)
        
##UNSUPERVISED ALGORITHMS######
    
    #DBSCAN Clustering
    def unsupervisedAlgos(self, xlist, n):
        
        algoList = ["unsupervised","Silhouette Score","DBSCAN","K-Means"]
        
        dbscan = self.dbscan(xlist, n)
        kmeans = self.kmeans(xlist, n)
        
        sscores = [dbscan, kmeans]
        
        #plt.plot(algoList[1:], sscores, color = 'b', marker = 'o', markersize = 10, label = "Silhouette Score", linestyle = '--', linewidth = 3)
        df = pd.DataFrame({'Algorithm': algoList[2:], 'Silhouette Score': sscores})
        splot = sns.barplot(x = 'Algorithm', y = 'Silhouette Score', data = df)
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
        plt.title("Silhouette Score Comparison Graph")
        plt.savefig("./static/images/unsupervised/comparison.png", bbox_inches = 'tight')
        plt.clf()
        
        return max(sscores),algoList[sscores.index(max(sscores))+2],sscores,algoList,len(algoList)
    
        
        
        
        