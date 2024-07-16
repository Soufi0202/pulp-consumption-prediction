import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
from PIL import Image

# Initialize session state for page navigation and authentication
if "page" not in st.session_state:
    st.session_state.page = "login"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Dummy user credentials
USER_CREDENTIALS = {
    "dounia": "dounia019",
}

def display_login():
    # Load logos and display with rounded borders
    university_logo = Image.open("UM6P_Logo.jpg")
    school = Image.open("OCP-logo.jpg")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        st.image(school, use_column_width=True)
    with col1:
        st.image(university_logo, use_column_width=True)
    st.markdown(
        """
        <style>
        .title {
            font-size: 52px;
            font-weight: bold;
            color: #4CAF50; /* Green */
            text-align: center;
            margin-bottom: 20px;
            margin-left: 25px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5); /* Shadow */
        }
        .label {
            font-size: 23px;
            color: #4CAF50; /* Green */
            font-weight: bold;
            margin-bottom:1px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5); /* Shadow */
        }
        .stTextInput>div>div>input {
            padding: 5px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 15px 32px;
            font-size: 29px;
            border-radius: 12px;
            border: none;
            display: block;
            margin: 15px auto;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">Login</h1>', unsafe_allow_html=True)

    st.markdown('<h5 class="label">Username : </h5>', unsafe_allow_html=True)
    username = st.text_input("")
    st.markdown('<h5 class="label">Password : </h5>', unsafe_allow_html=True)
    password = st.text_input("", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.page = "welcome"
            st.experimental_rerun()  # Refresh the page to go to the welcome page
        else:
            st.error("Invalid username or password")

def display_welcome():
    st.markdown(
        """
        <style>
        .title {
            font-size: 44px;
            font-weight: bold;
            color: #4CAF50; 
            text-align:center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0,0,0.5);
        }
        .title_p {
            font-size: 33px;
            font-weight: bold;
            color: #4CAF50; 
            margin-bottom: 14px;
            text-shadow: 0px 1px 4px rgba(0, 0,0,0.5);
        }
        .titlee {
            font-size: 22px;
            font-weight: bold;
            color: black; 
            text-align:center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0,0,0.5);
        }

        .u_title {
            font-size: 18px;
            font-weight: bold;
            color: black; 
        }

        .uu_title {
            font-size: 16px;
            color: black; 
        }
        .description {
            font-family: 'Times New Roman';
            font-weight: bold;
            font-size: 34px;
            text-align: justify;
            margin-top: 20px;
        }
        .description_p {
            font-size: 18px;
            font-weight: bold;
            color: black; 
            margin-top: 6px;
        }
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #3b8eda;
            text-align: center;
        }
        .btn-container {
            display:flex;
            justify-content: center;
            margin-top:30px;
        }
        .stButton button {
            background-color: #4CAF50;
            color:white;
            padding: 15px 32px;
            font-size: 20px;
            border-radius: 12px;
            border: none;
            display: block;
            margin: 0 auto;
        }
        .center-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 15px;
            overflow:hidden;
        }
        .green-line {
            width: 50%;
            height: 2px;
            background-color: #4CAF50;
        }
        .name {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        .name span {
            color: black;
        }
        img {
            border-radius: 15px;
        }
        .data-table {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .plot-container {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">Welcome to the Pulp Consumption Prediction Project</h1>', unsafe_allow_html=True)
    
    image_path = "pipe.jpg"
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    st.markdown('<h2 class="titlee">Une succession d’impacts positifs</h2>',unsafe_allow_html=True)
    st.markdown('<h3 class="u_title">OCP a été créé en 1920 en tant que Office Chérifien des Phosphates. Nous avons démarré notre activité avec l’exploitation d’une première mine à Khouribga. Nos activités s’étendent aujourd’hui sur cinq continents et nous travaillons tout au long de la chaîne de valeur des phosphates que ce soit dans l’extraction minière, la transformation industrielle ou encore l’éducation et le développement de communautés.</h3>',unsafe_allow_html=True)
    st.markdown('<h3 class="uu_title">OCP a démarré sa production en mars 1921 à Khouribga et exporté ses premiers produits depuis le port de Casablanca plus tard la même année. Une deuxième mine a été ouverte à Youssoufia en 1931 ainsi qu’une troisième plus tard en 1976 à Benguerir. Le Groupe OCP s’est ensuite diversifié en investissant dans la transformation des phosphates et en implantant des sites chimiques à Safi (1965) et Jorf Lasfar (1984).En 2008, l’Office Chérifien des Phosphates est devenu OCP Group S.A., propriété de l’Etat marocain et du Groupe Banque Populaire. Notre success-story a renforcé notre relation avec nos communautés, ancré notre engagement à réduire l’impact de nos activités sur l’environnement et motivé nos partenariats avec des entreprises locales et internationales innovantes.</h3>',unsafe_allow_html=True)
    image_path = "History_Copy.png"
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    st.markdown("""
    <div class="description">
        <div class="description">
            "Notre plus grand atout n'est pas le phosphate, mais l’Humain."
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="green-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="name"><span>--- Dr. Mostafa Terrab ---</span></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="title_p">Project Description</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main">
        <div class="description_p">
            This project aims to help OCP predict the amount of pulp consumption requested by clients.
        </div>
    </div>
    """, unsafe_allow_html=True)
    image_path = "deep.jpg"
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

    st.markdown("""
    <div class="main">
        <div class="description_p">
            Having a predicted value of the amount wanted by the client before the request will help in managing stocks
            and predicting shipments, ensuring efficient service to multiple clients.
        </div>
    """, unsafe_allow_html=True)

    data_logo = Image.open("data.jpg")
    ai_logo = Image.open("ai.jpg")
    pred_logo = Image.open("pred.jpg")

    col1, col2= st.columns([1,3])
    with col1:
        st.image(data_logo, use_column_width=True)
    with col2:
        st.markdown('<h3 class="uu_title">The data was sourced from Flux Department and included information such as pulp consumption volumes, client details, and timestamps. The data underwent several preprocessing steps including handling missing values, transforming categorical features, and engineering new features like holidays and time-based attributes.</h3>',unsafe_allow_html=True)
    

    st.markdown('<div class="green-line"></div>', unsafe_allow_html=True)

    col1, col2= st.columns([3,1])
    with col2:
        st.image(ai_logo, use_column_width=True)
    with col1:
        st.markdown('<h3 class="uu_title"> The primary objective was to develop a robust machine learning model to predict pulp consumption based on the preprocessed data. We experimented with several algorithms, ultimately selecting a RandomForestRegressor due to its superior performance. The model was trained on historical data and evaluated using metrics such as Mean Squared Error (MSE) to ensure its accuracy and reliability. </h3>',unsafe_allow_html=True)
    
    st.markdown('<div class="green-line"></div>', unsafe_allow_html=True)
    
    col1, col2= st.columns([1,3])
    with col1:
        st.image(pred_logo, use_column_width=True)
    with col2:
        st.markdown('<h3 class="uu_title"> After training the model, we used it to predict future pulp consumption for different clients. The predictions were visualized through various plots to help stakeholders understand the anticipated demand. These visualizations aid in effective stock management and planning for future shipments, ensuring that client demands are met efficiently.</h3>',unsafe_allow_html=True)

    st.markdown('<h2 class="title">Enjoy !</h2>', unsafe_allow_html=True)

def display_home():
    st.markdown(
        """
        <style>
        .title {
            font-size: 44px;
            font-weight: bold;
            color: #4CAF50; /* Green */
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5); /* Shadow */
        }
        .data {
            font-size: 22px;
            font-weight: bold;
            color: #4CAF50; /* Green */
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5); /* Shadow */
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        }
        .centered {
            display: flex;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">PULP Consumption Prediction App</h1>', unsafe_allow_html=True)

    # Load logos and display with rounded borders
    ocp_logo = Image.open("OCP-logo.jpg")
    university_logo = Image.open("UM6P_Logo.jpg")
    school = Image.open("cc-logo.png")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(ocp_logo, use_column_width=True)
    with col1:
        st.image(school, use_column_width=True)
    with col3:
        st.image(university_logo, use_column_width=True)

    # Upload Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file:
        # Load the data
        data = pd.read_excel(uploaded_file, sheet_name="Feuil1")
        st.markdown('<h2 class="data">Data from the uploaded Excel file:</h2>', unsafe_allow_html=True)
        st.dataframe(data)

        # Data cleaning and preparation
        data_cleaned = data.drop(columns=["QS"])
        data_cleaned = data_cleaned.dropna(subset=["VolPulpe", "Ton_Sec"])
        numeric_features = data_cleaned.select_dtypes(include=["float64", "int64"]).columns
        imputer_numeric = SimpleImputer(strategy="median")
        data_cleaned[numeric_features] = imputer_numeric.fit_transform(data_cleaned[numeric_features])
        categorical_features = data_cleaned.select_dtypes(include=["object"]).columns
        imputer_categorical = SimpleImputer(strategy="most_frequent")
        data_cleaned[categorical_features] = imputer_categorical.fit_transform(data_cleaned[categorical_features])
        data_cleaned["DAT"] = pd.to_datetime(data_cleaned["DAT"])
        data_cleaned["hour"] = data_cleaned["DAT"].dt.hour
        data_cleaned["day_of_week"] = data_cleaned["DAT"].dt.dayofweek
        data_cleaned["month"] = data_cleaned["DAT"].dt.month

        # Define fixed holidays
        fixed_holidays = ["01-01", "11-01", "05-01", "07-30", "08-14", "08-20", "08-21", "11-06", "11-18"]
        data_cleaned["is_holiday"] = data_cleaned["DAT"].apply(lambda x: 1 if x.strftime('%m-%d') in fixed_holidays else 0)
        fig,ax = plt.subplots(figsize=(14,7))
        sns.lineplot(data=data_cleaned, x="DAT", y="VolPulpe", hue="Client", ax=ax)
        ax.set_title("Consommation de pulpe par heure pour différents clients")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume de pulpe")
        ax.legend(title="Client")
        st.pyplot(fig)

        # Analyze daily and hourly trends
        data_cleaned["hour"] = data_cleaned["DAT"].dt.hour
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=data_cleaned, x="hour", y="VolPulpe", hue="Client", ax=ax)
        ax.set_title("Tendances horaires de la consommation de pulpe")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Volume de pulpe")
        ax.legend(title="Client")
        st.pyplot(fig)
        data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_features, drop_first=True)

        # Plot settings
        st.markdown('<h2 class="data">Relationship between VolPulpe and other features:</h2>', unsafe_allow_html=True)
        features = data_cleaned.columns.drop(["DAT", "VolPulpe"])
        sns.set_theme(style="darkgrid")
        colors = ['#FF6347', '#4682B4', '#32CD32', '#8A2BE2', '#FFD700', '#FF69B4', '#D2691E', '#6495ED', '#DC143C', '#FF8C00']

        def describe_relation(feature, target):
            correlation = np.corrcoef(data_cleaned[feature], data_cleaned[target])[0, 1]
            description = f"Correlation between {feature} and {target}: {correlation:.2f}. "

            if correlation > 0.5:
                description += "This means that as one increases, the other tends to increase significantly."
            elif correlation < -0.5:
                description += "This means that as one increases, the other tends to decrease significantly. "
            elif correlation > 0.3:
                description += "This indicates a moderate positive relationship. "
            elif correlation < -0.3:
                description += "This indicates a moderate negative relationship. "
            else:
                description += "This indicates a weak or no clear relationship. For example, if the correlation is 0.1 or -0.1, the feature doesn't have much of an effect on the pulp volume."

            return description


        plots = []
        for i, feature in enumerate(features):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data_cleaned, x=feature, y="VolPulpe", ax=ax, color=colors[i % len(colors)])
            ax.set_title(f"VolPulpe vs {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("VolPulpe")
            description = describe_relation(feature, "VolPulpe")
            ax.text(0.5, -0.2, description, transform=ax.transAxes, ha='center', va='center', fontsize=10, color='green')
            plots.append(fig)

        if 'plot_index' not in st.session_state:
            st.session_state.plot_index = 0

        col1, col2, col3 = st.columns([2, 8, 2])

        if col1.button("Previous"):
            st.session_state.plot_index = max(0, st.session_state.plot_index - 1)

        if col3.button("Next"):
            st.session_state.plot_index = min(len(plots) - 1, st.session_state.plot_index + 1)

        st.pyplot(plots[st.session_state.plot_index])

        # Client and precision selection
        clients = [col for col in data_cleaned.columns if col.startswith('Client_')]
        client_selection = st.selectbox("Select a client to predict consumption", clients)
        precision_options = {"Low . 1-5 minutes": (0.1, 3), "Medium . 5-20 minutes": (0.05, 5), "High . 20-60 minutes": (0.01, 10)}
        precision_selection = st.selectbox("Select precision level", list(precision_options.keys()))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            predict_daily = st.button("Predict Daily")
        with col3:
            predict_hourly = st.button("Predict Hourly")

        if predict_daily:
            def train_and_predict_for_client(client, data, precision):
                client_data = data[data[client] == 1]
                if client_data.empty:
                    return pd.DataFrame()

                X = client_data.drop(columns=['DAT', 'VolPulpe'])
                y = client_data['VolPulpe']
                X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

                rf = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                best_rf = grid_search.best_estimator_
                y_val_pred = best_rf.predict(X_val)
                mse_val = mean_squared_error(y_val, y_val_pred)

                num_days_to_predict = len(client_data)
                future_dates = pd.date_range(start=client_data['DAT'].max() + pd.Timedelta(days=1), periods=num_days_to_predict)
                last_known_data = client_data.drop(columns=['DAT', 'VolPulpe']).iloc[-1].values.reshape(1, -1)
                predictions = []

                for future_date in future_dates:
                    next_day_prediction = best_rf.predict(last_known_data)
                    predictions.append((future_date, next_day_prediction[0]))
                    last_known_data = np.roll(last_known_data, -1)
                    last_known_data[0, -1] = next_day_prediction[0]

                prediction_df = pd.DataFrame(predictions, columns=['DAT', 'VolPulpe'])
                return prediction_df

            precision_value, n_estimators = precision_options[precision_selection]
            client_predictions = train_and_predict_for_client(client_selection, data_cleaned, precision_value)
            st.session_state.daily_predictions = client_predictions
            


        if predict_hourly:
            def predict_hourly_for_client(client, data, precision):
                client_data = data[data[client] == 1]
                if client_data.empty:
                    return pd.DataFrame()

                X = client_data.drop(columns=['DAT', 'VolPulpe'])
                y = client_data['VolPulpe']
                X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

                rf = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                best_rf = grid_search.best_estimator_
                y_val_pred = best_rf.predict(X_val)
                mse_val = mean_squared_error(y_val, y_val_pred)

                num_hours_to_predict = len(client_data)
                future_dates = pd.date_range(start=client_data['DAT'].max() + pd.Timedelta(hours=1), periods=num_hours_to_predict, freq='H')
                last_known_data = client_data.drop(columns=['DAT', 'VolPulpe']).iloc[-1].values.reshape(1, -1)
                predictions = []

                for future_date in future_dates:
                    next_hour_prediction = best_rf.predict(last_known_data)
                    predictions.append((future_date, next_hour_prediction[0]))
                    last_known_data = np.roll(last_known_data, -1)
                    last_known_data[0, -1] = next_hour_prediction[0]

                prediction_df = pd.DataFrame(predictions, columns=['DAT', 'VolPulpe'])
                return prediction_df

            precision_value, n_estimators = precision_options[precision_selection]
            client_hourly_predictions = predict_hourly_for_client(client_selection, data_cleaned, precision_value)
            st.session_state.hourly_predictions = client_hourly_predictions
            

    else:
        st.warning("Please upload an Excel file to proceed.")

def display_results():
    st.markdown(
        """
        <style>
        .title {
            font-size: 44px;
            font-weight: bold;
            color: #4CAF50; 
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5);
        }
        .data {
            font-size: 22px;
            font-weight: bold;
            color: #4CAF50; 
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.5);
        }
        .plot-container {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            color: black;
            text-align: justify;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        }
        .titleee {
            font-size: 22px;
            font-weight: bold;
            color: #FF0000; 
            text-align:center;
            margin-bottom: 20px;
            text-shadow: 0px 1px 4px rgba(0, 0,0,0.5);
        }

        .disc {
            font-size: 18px;
            font-weight: bold;
            color: black; 
            margin-bottom: 30px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<h2 class="title">Those are the Final Predictions:</h2>', unsafe_allow_html=True)

    if 'daily_predictions' in st.session_state:
        st.markdown('<h3 class="data">Daily Predictions :</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])  # Adjusted column width ratios
        
        with col1:
            st.write(st.session_state.daily_predictions)
        with col2:
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased plot size
            ax.plot(st.session_state.daily_predictions['DAT'], st.session_state.daily_predictions['VolPulpe'], marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Volume de pulpe")
            ax.set_title("Daily Predictions")
            st.pyplot(fig)

    if 'hourly_predictions' in st.session_state:
        st.markdown('<h3 class="data">Hourly Predictions :</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])  # Adjusted column width ratios
        
        with col1:
            st.write(st.session_state.hourly_predictions)
        with col2:
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased plot size
            ax.plot(st.session_state.hourly_predictions['DAT'], st.session_state.hourly_predictions['VolPulpe'], marker='o')
            ax.set_xlabel("Date and Hour")
            ax.set_ylabel("Volume de pulpe")
            ax.set_title("Hourly Predictions")
            st.pyplot(fig)
    st.markdown('<h2 class="titleee">IMPORTANT !</h2>',unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disc">
        <strong>Disclaimer:</strong> The results presented here are predictions and do not guarantee the exact amounts that clients will request. These values are subject to change based on various factors including client behavior and external conditions.
        </div>
        """, unsafe_allow_html=True
    )

# Display content based on selected page
if not st.session_state.authenticated:
    display_login()
else:
    if st.session_state.page == "home":
        display_home()
    elif st.session_state.page == "results":
        display_results()
    elif st.session_state.page == "welcome":
        display_welcome()

# Navigation buttons at the bottom
if st.session_state.authenticated:
    col1, _, col2, _, col3 = st.columns([3, 3, 3, 3, 3])

    if col1.button("Welcome"):
        st.session_state.page = "welcome"
    if col2.button("Home"):
        st.session_state.page = "home"
    if col3.button("Results"):
        st.session_state.page = "results"
