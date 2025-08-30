24import joblib
import json
import pandas as pd
import argparse
import sys


class SocialMediaAddictionPredictor:
    def __init__(self, model_path='social_media_addiction_model.pkl',
                 scaler_path='scaler.pkl',
                 metadata_path='model_metadata.json'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            self.feature_names = self.metadata['feature_names']
            self.numerical_features = self.metadata['numerical_features']
            self.categorical_mappings = self.metadata['categorical_mappings']
            self.one_hot_columns = self.metadata['one_hot_columns']

            print("Model loaded successfully!")
            print(f"Model Performance: R² = {self.metadata['model_performance']['r2_score']:.4f}")

        except FileNotFoundError as e:
            print(f"Error: Could not load model files. {e}")
            print("Make sure you have run the training script first to generate the model files.")
            sys.exit(1)

    def get_user_input_interactive(self):
        print("\n" + "=" * 60)
        print("SOCIAL MEDIA ADDICTION SCORE PREDICTOR")
        print("=" * 60)
        print("Please provide the following information:")

        user_data = {}

        valid_options = {
            'Gender': ['Female', 'Male'],
            'Academic_Level': ['Undergraduate', 'Graduate', 'High School'],
            'Country': ['Bangladesh', 'India', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'Brazil',
                        'Japan', 'South Korea', 'France', 'Spain', 'Italy', 'Mexico', 'Russia', 'China',
                        'Sweden', 'Norway', 'Denmark', 'Netherlands', 'Belgium', 'Switzerland',
                        'Austria', 'Portugal', 'Greece', 'Ireland', 'New Zealand', 'Singapore',
                        'Malaysia', 'Thailand', 'Vietnam', 'Philippines', 'Indonesia', 'Taiwan',
                        'Hong Kong', 'Turkey', 'Israel', 'UAE', 'Egypt', 'Morocco', 'South Africa',
                        'Nigeria', 'Kenya', 'Ghana', 'Argentina', 'Chile', 'Colombia', 'Peru',
                        'Venezuela', 'Ecuador', 'Uruguay', 'Paraguay', 'Bolivia', 'Costa Rica',
                        'Panama', 'Jamaica', 'Trinidad', 'Bahamas', 'Iceland', 'Finland', 'Poland',
                        'Romania', 'Hungary', 'Czech Republic', 'Slovakia', 'Croatia', 'Serbia',
                        'Slovenia', 'Bulgaria', 'Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Moldova',
                        'Belarus', 'Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan', 'Armenia',
                        'Georgia', 'Azerbaijan', 'Cyprus', 'Malta', 'Luxembourg', 'Monaco', 'Andorra',
                        'San Marino', 'Vatican City', 'Liechtenstein', 'Montenegro', 'Albania',
                        'North Macedonia', 'Kosovo', 'Bosnia', 'Qatar', 'Kuwait', 'Bahrain', 'Oman',
                        'Jordan', 'Lebanon', 'Iraq', 'Yemen', 'Syria', 'Afghanistan', 'Pakistan',
                        'Nepal', 'Bhutan', 'Sri Lanka', 'Maldives'],
            'Most_Used_Platform': ['Instagram', 'Twitter', 'TikTok', 'YouTube', 'Facebook', 'LinkedIn',
                                   'Snapchat', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'],
            'Relationship_Status': ['In Relationship', 'Single', 'Complicated'],
            'Affects_Academic_Performance': ['Yes', 'No']
        }

        user_data['Age'] = float(input("Age: "))
        user_data['Avg_Daily_Usage_Hours'] = float(input("Average Daily Social Media Usage (hours): "))
        user_data['Sleep_Hours_Per_Night'] = float(input("Sleep Hours Per Night: "))
        user_data['Mental_Health_Score'] = float(input("Mental Health Score (1-10): "))
        user_data['Conflicts_Over_Social_Media'] = float(input("Conflicts Over Social Media (scale 1-10): "))

        print("\nCategorical Information:")

        while True:
            print(f"Academic Level options: {', '.join(valid_options['Academic_Level'])}")
            user_data['Academic_Level'] = input("Academic Level: ").strip()
            if user_data['Academic_Level'] in valid_options['Academic_Level']:
                break
            print("Invalid option. Please choose from the available options.")

        while True:
            print(f"Gender options: {', '.join(valid_options['Gender'])}")
            user_data['Gender'] = input("Gender: ").strip()
            if user_data['Gender'] in valid_options['Gender']:
                break
            print("Invalid option. Please choose from the available options.")

        while True:
            print(
                "Available countries (showing first 20): Bangladesh, India, USA, UK, Canada, Australia, Germany, Brazil, Japan, South Korea, France, Spain, Italy, Mexico, Russia, China, Sweden, Norway, Denmark, Netherlands... (and many more)")
            user_data['Country'] = input("Country: ").strip()
            if user_data['Country'] in valid_options['Country']:
                break
            print("Invalid country. Please enter a valid country name from the dataset.")

        while True:
            print(f"Platform options: {', '.join(valid_options['Most_Used_Platform'])}")
            user_data['Most_Used_Platform'] = input("Most Used Social Media Platform: ").strip()
            if user_data['Most_Used_Platform'] in valid_options['Most_Used_Platform']:
                break
            print("Invalid platform. Please choose from the available options.")

        while True:
            print(f"Relationship Status options: {', '.join(valid_options['Relationship_Status'])}")
            user_data['Relationship_Status'] = input("Relationship Status: ").strip()
            if user_data['Relationship_Status'] in valid_options['Relationship_Status']:
                break
            print("Invalid option. Please choose from the available options.")

        while True:
            print(
                f"Does social media affect academic performance? ({', '.join(valid_options['Affects_Academic_Performance'])})")
            user_data['Affects_Academic_Performance'] = input("Affects Academic Performance: ").strip()
            if user_data['Affects_Academic_Performance'] in valid_options['Affects_Academic_Performance']:
                break
            print("Invalid option. Please enter Yes or No.")

        return user_data

    def preprocess_input(self, user_data):
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)

        for feature in self.numerical_features:
            if feature in user_data:
                input_df[feature] = user_data[feature]

        if user_data['Academic_Level'] in self.categorical_mappings['Academic_Level']:
            input_df['Academic_Level_encoded'] = self.categorical_mappings['Academic_Level'][
                user_data['Academic_Level']]

        if user_data['Affects_Academic_Performance'] in self.categorical_mappings['Affects_Academic_Performance']:
            input_df['Affects_Academic_Performance_encoded'] = \
            self.categorical_mappings['Affects_Academic_Performance'][user_data['Affects_Academic_Performance']]

        for category, columns in self.one_hot_columns.items():
            if category in user_data:
                user_value = user_data[category]
                for col in columns:
                    if col.endswith(f'_{user_value}'):
                        input_df[col] = 1
                        break

        input_df[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])

        return input_df

    def predict(self, user_data):
        processed_input = self.preprocess_input(user_data)
        prediction = self.model.predict(processed_input)[0]
        return prediction

    def interpret_score(self, score):
        if score < 3:
            level = "Low"
            description = "Low social media addiction risk"
        elif score < 5:
            level = "Moderate"
            description = "Moderate social media usage - be mindful of time spent"
        elif score < 7:
            level = "High"
            description = "High social media usage - consider reducing usage"
        else:
            level = "Very High"
            description = "Very high social media addiction risk - consider seeking help"

        return level, description


def main():
    parser = argparse.ArgumentParser(description='Predict Social Media Addiction Score')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--age', type=float, help='Age')
    parser.add_argument('--daily-usage', type=float, help='Average daily usage in hours')
    parser.add_argument('--sleep-hours', type=float, help='Sleep hours per night')
    parser.add_argument('--mental-health', type=float, help='Mental health score (1-10)')
    parser.add_argument('--conflicts', type=float, help='Conflicts over social media (1-10)')
    parser.add_argument('--academic-level', type=str, help='Academic level')
    parser.add_argument('--gender', type=str, help='Gender')
    parser.add_argument('--country', type=str, help='Country')
    parser.add_argument('--platform', type=str, help='Most used platform')
    parser.add_argument('--relationship', type=str, help='Relationship status')
    parser.add_argument('--affects-academic', type=str, help='Affects academic performance (Yes/No)')

    args = parser.parse_args()

    predictor = SocialMediaAddictionPredictor()

    if args.interactive or not any(vars(args).values()):
        user_data = predictor.get_user_input_interactive()
    else:
        user_data = {
            'Age': args.age,
            'Avg_Daily_Usage_Hours': args.daily_usage,
            'Sleep_Hours_Per_Night': args.sleep_hours,
            'Mental_Health_Score': args.mental_health,
            'Conflicts_Over_Social_Media': args.conflicts,
            'Academic_Level': args.academic_level,
            'Gender': args.gender,
            'Country': args.country,
            'Most_Used_Platform': args.platform,
            'Relationship_Status': args.relationship,
            'Affects_Academic_Performance': args.affects_academic
        }

        if None in user_data.values():
            print("Error: All arguments are required for non-interactive mode.")
            print("Use --interactive flag for interactive input or provide all arguments.")
            sys.exit(1)

    try:
        predicted_score = predictor.predict(user_data)
        level, description = predictor.interpret_score(predicted_score)

        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Addiction Score: {predicted_score:.2f}")
        print(f"Addiction Level: {level}")
        print(f"Interpretation: {description}")
        print("=" * 60)

    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please check your input values and try again.")


if __name__ == "__main__":
    main()