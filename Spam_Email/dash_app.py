import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import joblib

# --- 1. LOAD THE PIPELINE ---
try:
    # We only load one file now!  
    pipeline = joblib.load(r'C:\Users\USER\Desktop\100 Days ML\Spam_email_Classifier\Spam_Email\Spam_Email_Classifier_Model.pkl')
    print("Pipeline loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load 'Spam_Email_Classifier_Model'. {e}")
    pipeline = None
    exit

# --- 2. SETUP THE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG]) # Dark mode

app.layout = dbc.Container([
    
    html.H1("📧 Spam Filter AI", className="text-center my-4 text-light"),

    dbc.Row([
        # INPUT COLUMN
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Step 1: Paste Email Text"),
                dbc.CardBody([
                    dcc.Textarea(
                        id='email-input',
                        placeholder='Type or paste email content here...',
                        value='Congratulations! You have won free money.', 
                        style={'width': '100%', 'height': '200px', 'color': 'black'}
                    ),
                ])
            ], color="secondary", inverse=True)
        ], width=6),

        # OUTPUT COLUMN
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Step 2: AI Diagnosis"),
                dbc.CardBody([
                    html.H2(id='prediction-text', className="text-center display-4 bold"),
                    html.Hr(),
                    html.Label("Confidence Score:"),
                    dbc.Progress(id="confidence-bar", value=0, striped=True, animated=True),
                ])
            ], id="result-card", color="light", inverse=False)
        ], width=6)
    ])

], fluid=True)

# --- 3. THE LOGIC ---
@app.callback(
    [Output('prediction-text', 'children'),
     Output('result-card', 'color'),   
     Output('result-card', 'inverse'), 
     Output('confidence-bar', 'value'),
     Output('confidence-bar', 'color')],
    [Input('email-input', 'value')]
)
def predict_spam(email_text):
    if not email_text or pipeline is None:
        return "Waiting...", "light", False, 0, "info"

    try:
        # NOTICE: We pass the text directly to the pipeline!
        # The pipeline runs Tfidf -> Logistic Regression automatically.
        prediction = pipeline.predict([email_text])[0]
        
        # Get probability (returns [[prob_ham, prob_spam]])
        probs = pipeline.predict_proba([email_text])[0]
        prob_spam = probs[1] * 100  # Index 1 is Spam
        prob_ham = probs[0] * 100   # Index 0 is Ham
        
        # DECISION LOGIC
        if prediction == 1: # 1 = Spam
            return "🚨 SPAM DETECTED", "danger", True, prob_spam, "danger"
        else:
            return "✅ SAFE EMAIL", "success", True, prob_ham, "success"

    except Exception as e:
        return f"Error: {str(e)}", "warning", False, 0, "warning"

if __name__ == '__main__':
    app.run(debug=True)