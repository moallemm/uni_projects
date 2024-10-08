from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)

# Configure MySQL database path
username = 'root'
password = 'M@allem369'
host = 'localhost'
port = '3306'
database_name = 'crafts_db'

app.secret_key = 'it_is_a_secret'  #secret key for flash messages

# Encode password for URI
encoded_password = urllib.parse.quote_plus(password)
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql://{username}:{encoded_password}@{host}:{port}/{database_name}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initializing SQLAlchemy instance
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Specifying the login view

# Define the Craft table model
class Craft(db.Model):
    __tablename__ = 'crafts'  #specifying the table name to avoid conflicts
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    image_url = db.Column(db.Text, nullable=False)
    video_url = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'Craft(id={self.id}, name={self.name})'
# Define the user table model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(100), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define the Feedback table model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'Feedback(id={self.id}, name={self.name})'
#More info page
@app.route('/craftinfo/<int:craft_id>')
def info(craft_id):
    craft = Craft.query.get_or_404(craft_id)
    return render_template('InfoPage.html', craft=craft)
#contact info page
@app.route('/contactinfo')
def contactpage():
    return render_template('AboutUs.html')
#when we click the submit feedback this route will work
@app.route('/submitfeedback')
def feed():
    return render_template('Feedback.html')
#this specifies the functionality of the feedback page
@app.route('/feedback', methods=['GET', 'POST'])
def submit_feedback():
    if request.method == 'POST':
        name = request.form['username']
        email = request.form['email']
        message = request.form['message']

        if not name or not email or not message:
            flash('Please fill out all fields', 'error')
        else:
            new_feedback = Feedback(name=name, email=email, message=message)
            db.session.add(new_feedback)
            db.session.commit()
            flash('Thank you for your feedback!', 'success')

    return redirect(url_for('homepage'))

# User Loader for Flask-Login jebneha mn chatgpt
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
#this specifies the functionality of the login page
@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Login successful
            if user.status == "Approved":
                login_user(user)  # Use login_user to handle session management
                flash('Login successful!', 'success')
                if user.role == "admin":
                    return redirect(url_for('Admin'))
                else:
                    return redirect(url_for('homepage'))
            elif user.status == "Banned":
                flash('Your account has been banned!', 'error')
        else:
            # Invalid credentials
            flash('Invalid username or password', 'error')
    return render_template('Login.html')

#this is the Logout Route when we press any logout in the whole website this well work
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))
# this specifies the function of the register page
@app.route('/Register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        # Check if username is already taken or not
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        # Create new user if the user does not exist and the inputed name is not already taken
        new_user = User(username=username, role="user", status="Approved")
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('Register.html')


# Homepage route to display the crafts
@app.route('/HomePage')
@login_required
def homepage():
    # Retrieve recently added crafts this data takes the last 6 which are the recenttl added
    recently_added_crafts = Craft.query.order_by(Craft.id.desc()).limit(6).all()

    # Retrieve most trending crafts this does not work we did not have time it was supposed to display the top 6 with most likes
    most_trending_crafts = Craft.query.order_by(Craft.id.desc()).limit(6).all()

    # Retrieve recently searched crafts
    search_term = request.args.get('search', '')
    top_searched_crafts = Craft.query.filter(Craft.name.ilike(f'%{search_term}%')).limit(3).all()

    return render_template('HomePage.html', 
                           recently_added=recently_added_crafts,
                           most_trending=most_trending_crafts,
                           top_searched=top_searched_crafts)
# render the intro page only
@app.route('/')
def mainpage():
    return render_template('intro.html')
# specifies the search functionality of all the search boxs
@app.route('/search', methods=['GET'])
def search_crafts():
    search_term = request.args.get('search', '')
    if search_term:
        # Query crafts whose name contains the search term
        matched_crafts = Craft.query.filter(Craft.name.ilike(f'%{search_term}%')).all()
    else:
        matched_crafts = []  # No search term provided, return empty list

    return render_template('Search.html', crafts=matched_crafts, search=search_term)
# this function is for the admin dashboard page
@app.route('/Admin', methods=['GET'])
@login_required
def Admin():
    if current_user.role != 'admin':
        abort(403)  # Forbidden if the role is not admin he will not be granted access
        
    crafts_list = Craft.query.all()
    feedback_list = Feedback.query.all()
    user_list = User.query.all()
    return render_template('Admin.html', crafts=crafts_list, feedbacks=feedback_list, users=user_list)
#add craft for the admin
@app.route('/add_craft', methods=['POST'])
@login_required
def add_craft():
    name = request.form['name']
    image_url = request.form['image_url']
    video_url = request.form['video_url']
    description = request.form['description']
    
    new_craft = Craft(name=name, image_url=image_url, video_url=video_url, description=description)
    db.session.add(new_craft)
    db.session.commit()

    return redirect(url_for('Admin'))
#edit craft for the admin
@app.route('/edit_craft', methods=['POST'])
@login_required
def edit_craft():
    craft_id = request.form['id']
    name = request.form['name']
    image_url = request.form['image_url']
    video_url = request.form['video_url']
    description = request.form['description']
    
    # Retrieve craft data from the database
    craft = Craft.query.get(craft_id)

    if craft:
        # change the craft attributes
        craft.name = name
        craft.image_url = image_url
        craft.video_url = video_url
        craft.description = description

        # Commit the changes to the database
        db.session.commit()

        flash('Craft updated successfully!', 'success')
    else:
        flash('Craft not found.', 'error')

    return redirect(url_for('Admin'))
#delete craft for the admin
@app.route('/delete_craft/<int:craft_id>', methods=['POST', 'DELETE'])
@login_required
def delete_craft(craft_id):
    if request.method == 'POST' or request.method == 'DELETE':
        # Retrieve craft data from the database
        craft = Craft.query.get(craft_id)

        if craft:
            # Delete the craft from the database
            db.session.delete(craft)
            db.session.commit()

            flash('Craft deleted successfully!', 'success')
        else:
            flash('Craft not found.', 'error')

    return redirect(url_for('Admin'))
#delete craft feedback for the admin
@app.route('/delete_feedback/<int:feedback_id>', methods=['POST', 'DELETE'])
@login_required
def delete_feedback(feedback_id):
    if request.method == 'POST' or request.method == 'DELETE':
        feedback = Feedback.query.get_or_404(feedback_id)
        if feedback:
            db.session.delete(feedback)
            db.session.commit()
            flash('Feedback deleted successfully!', 'success')
        else:
            flash('Feedback not found.', 'error')

    return redirect(url_for('Admin'))  # Redirect to admin page after deletion
#ban users for the admin
@app.route('/ban_user/<int:user_id>', methods=['POST', 'DELETE'])
@login_required
def ban_user(user_id):
    if request.method == 'POST' or request.method == 'DELETE':
        # take user from the database
        user_to_ban = User.query.get_or_404(user_id)

        if user_to_ban:
            # Check if the user to be banned is a user not an since admins can not be banned
            if user_to_ban.role == 'user':
                # change the user's status to 'banned'
                user_to_ban.status = 'banned'
                db.session.commit()
                flash('User banned successfully!', 'success')
            elif user_to_ban.role == 'admin':
                flash('Admin user cannot be banned.', 'error')
            else:
                flash('Invalid user role.', 'error')
        else:
            flash('User not found.', 'error')

    return redirect(url_for('Admin'))  # Redirect to admin page after banning
#unban user for admin
@app.route('/unban_user/<int:user_id>', methods=['POST'])
@login_required
def unban_user(user_id):
    # Fetch the user from the database
    user_to_unban = User.query.get_or_404(user_id)

    if user_to_unban:
        # Check if the user is banned and can be unbanned
        if user_to_unban.status == 'banned':
            # change the user's status to 'active'
            user_to_unban.status = 'Approved'
            db.session.commit()
            flash('User unbanned successfully!', 'success')
        else:
            flash('User is not currently banned.', 'error')
    else:
        flash('User not found.', 'error')

    return redirect(url_for('Admin'))  # Redirect to admin page after unbanning

if __name__ == '__main__':
    app.run(debug=True)
