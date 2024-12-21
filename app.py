from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(_name_)

# Load the trained model and vectorizer
with open('fake_job_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home route to render the HTML form and handle predictions."""
    description = None
    prediction = None
    prediction_color = None
    if request.method == 'POST':
        # Get the job description from the form
        description = request.form['description']
        
        # Transform the input using the loaded vectorizer
        input_tfidf = tfidf_vectorizer.transform([description])
        
        # Make a prediction using the trained model
        prediction = model.predict(input_tfidf)
        
        # Map prediction to label and set color
        if prediction[0] == 1:
            prediction = "Fake Job Posting"
            prediction_color = "red"
        else:
            prediction = "Real Job Posting"
            prediction_color = "green"
    
    return render_template('index.html', description=description, prediction=prediction, prediction_color=prediction_color)

# Run the app
if _name_ == '_main_':
    app.run(debug=True)
