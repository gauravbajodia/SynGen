from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from werkzeug.utils import secure_filename
from SynGen import generate_from_scratch, generate_from_data, compare_algo
from flask.helpers import send_file
import os
import yaml
import pandas as pd
import shutil
from flask_sqlalchemy import SQLAlchemy

obj = generate_from_scratch()

app = Flask(__name__)
#db = SQLAlchemy(app)
mysql = MySQL(app)


#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://gligeftegfsedn:f60baa1c5e90155d15dc907db87d8bdbb9f60b8a5c215450e6caa8dcecb2e886@ec2-34-193-112-164.compute-1.amazonaws.com:5432/d60t4hpg6s87ot'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'DATABASE_URL'

#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# Enter your database connection details below

db = yaml.load(open('SQL_Login.yaml'))

app.config['MYSQL_HOST'] = db['MYSQL_HOST']
app.config['MYSQL_USER'] = db['MYSQL_USER']
app.config['MYSQL_PASSWORD'] = db['MYSQL_PASSWORD']
app.config['MYSQL_DB'] = db['MYSQL_DB']


'''class logindatabase(db.Model):
    __tablename__ = 'database'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), unique=True)
    password = db.Column(db.String(200),unique=True)
    email = db.Column(db.String(200),unique=True)

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
   '''     





app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.secret_key = '1234'

# http://localhost:5000/login - this will be the login page, we need to use both GET and POST requests


@app.route('/')
def gotoLogin():
    return redirect("/login")

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = 'An error occured'
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg='')

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/home',methods=['GET'])
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        if request.method == 'GET':
 
            clean_folders("./downloads")
            clean_folders("./uploads")
            clean_folders("./static/images/classifications")
            clean_folders("./static/images/regressions")
            clean_folders("./static/images/unsupervised")
             
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
def clean_folders(file_path):
   
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
        os.mkdir(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
        os.mkdir(file_path)

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# MODULE 1

@app.route('/module1', methods =["GET", "POST"])
def module1():
    
    if 'loggedin' in session:
        if request.method == 'GET':
            
            return render_template("module1.html") 
        if request.method == "POST": 
        # getting input with name = fields in HTML form 

            inputs = request.form.getlist('field[]')
            row_num = request.form.get('row_num')    #number of rows to be generated
            file_type = request.form.get('file_type')
            df = obj.gen_dataframe(fields=inputs,num=row_num)

            if(str(file_type) == "csv"):
                return send_file(obj.gen_csv(df),as_attachment='True')
            elif(str(file_type) == "excel"):
                return send_file(obj.gen_excel(df),as_attachment='True')
        
        return render_template("module1.html") 
    return redirect(url_for('login'))

# MODULE 2
@app.route('/module2', methods =["GET", "POST"])
def module2():
    if 'loggedin' in session:

        if request.method == 'GET':
            
            return render_template("module2.html") 
        if request.method == 'POST':
            f = request.files['file']   
            row_num = request.form.get('row_num')    #number of rows to be generated
            primary_key_index=request.form.get("primary_key_index")
            file_type = request.form.get('file_type')
            
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename("mod2_generated_synthetic_dataset.csv"))
            f.save(file_path)

            # Generate Dataset
            df=pd.read_csv(file_path) 
            Mod2 =  generate_from_data(df)
            generated=Mod2.learn_and_generate(pkey=primary_key_index,num=int(row_num))
            
            # Download Dataset
            if(str(file_type) == "csv"):
                return send_file(obj.gen_csv(generated),as_attachment='True')
            elif(str(file_type) == "excel"):
                return send_file(obj.gen_excel(generated),as_attachment='True')
            
        return render_template("module2.html") 
    return redirect(url_for('login'))



# MODULE 3
@app.route('/module3', methods =["GET", "POST"])
def module3():
    if 'loggedin' in session:
        if request.method == 'GET':
            return render_template("module3.html") 
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        x= request.form.get("x_value")
        y= request.form.get("y_value")
        algoType= request.form.get("type_select")

        result = []
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename("mod3_uploaded_dataset.csv"))
        f.save(file_path)

        df=pd.read_csv(file_path) 
        compareAlgo =  compare_algo(df)

        if(algoType=="supervised_learning"):
            result = compareAlgo.supervisedAlgos(x,y)  
        elif(algoType=="classification"):
            result =  compareAlgo.classification(x,y)    
        elif(algoType=="unsupervised_learning"):
            result =  compareAlgo.unsupervisedAlgos(x,int(y))
           
    return render_template("module3.html",result=result[0],mostEfficientAlgo=result[1],accuracies=result[2],algoList=result[3],len=result[4],img=1)

if __name__ == "__main__":
    app.run(debug=False)
