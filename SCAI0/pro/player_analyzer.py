import pandas as pd
import numpy as np
from enhanced_model import TalentScoutAI
from datetime import datetime
import os

class FootballScout:
    def __init__(self):
        self.model = TalentScoutAI()
        self.category_thresholds = None
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.all_players = self._load_all_players()
        self._initialize_model()

    def _load_all_players(self):
        """Load and validate all player data with enhanced category handling"""
        try:
            # Load data with explicit category labels and enhanced categorization
            middle = pd.read_csv(os.path.join(self.base_path, 'data', 'middle', 'middle.csv'))
            middle['category'] = 'MIDDLE'
            
            strong = pd.read_csv(os.path.join(self.base_path, 'data', 'strong', 'strong.csv'))
            strong['category'] = 'STRONG'
            
            weak = pd.read_csv(os.path.join(self.base_path, 'data', 'weak', 'weak.csv'))
            weak['category'] = 'WEAK'
            
            # Combine all data
            all_data = pd.concat([middle, strong, weak], ignore_index=True)
            
            # Enhanced category thresholds with stricter criteria
            self.category_thresholds = {
                'STRONG': {
                    'min_overall': 80,
                    'min_potential': 85,
                    'min_value': 1000000,
                    'avg_value': 2000000
                },
                'MIDDLE': {
                    'min_overall': 70,
                    'min_potential': 75,
                    'min_value': 500000,
                    'avg_value': 1000000
                },
                'WEAK': {
                    'max_overall': 69,
                    'max_potential': 74,
                    'max_value': 499999,
                    'avg_value': 250000
                }
            }
            
            # Clean and validate numeric columns with enhanced validation
            numeric_columns = ['Age', 'Overall', 'Potential', 'Value(£)', 'speed', 'dodge', 'Height(cm.)', 'Weight(lbs.)']
            for col in numeric_columns:
                all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
                all_data[col] = all_data[col].fillna(all_data[col].mean())  # Fill missing values with mean
                all_data[col] = all_data[col].abs()
            
            # Enhanced validation rules with stricter criteria
            all_data['Value(£)'] = all_data.apply(
                lambda row: self._validate_value(row), axis=1
            )
            
            all_data['Overall'] = all_data['Overall'].clip(1, 99)
            all_data['Potential'] = all_data['Potential'].clip(1, 99)
            
            # Recategorize players based on enhanced criteria
            all_data['category'] = all_data.apply(self._categorize_player, axis=1)
            
            return all_data
            
        except Exception as e:
            print(f"Error loading player data: {str(e)}")
            # Create a minimal dataset for testing
            return pd.DataFrame({
                'Age': [20], 'Overall': [75], 'Potential': [80],
                'Value(£)': [500000], 'speed': [70], 'dodge': [70],
                'Height(cm.)': [180], 'Weight(lbs.)': [170],
                'category': ['MIDDLE']
            })

    def _validate_value(self, row):
        """Validate and adjust player value based on category and stats with enhanced logic"""
        try:
            overall = row['Overall']
            potential = row['Potential']
            age = row['Age']
            
            # Base value calculation based on overall rating
            base_value = overall * 1000
            
            # Potential bonus
            potential_bonus = (potential - overall) * 500
            
            # Age factor - younger players get bonus
            age_factor = max(0.5, (30 - age) / 10) if age <= 30 else 0.3
            
            # Calculate final value
            calculated_value = (base_value + potential_bonus) * age_factor
            
            # Apply minimum and maximum thresholds
            if overall >= 80 or potential >= 85:  # Strong players
                calculated_value = max(calculated_value, 1000000)
            elif overall >= 70 or potential >= 75:  # Middle tier
                calculated_value = min(max(calculated_value, 500000), 999999)
            else:  # Weak players
                calculated_value = min(calculated_value, 499999)
                # Ensure very low-rated players get very low values
                if overall < 30:
                    calculated_value = min(calculated_value, 50000)
            
            return max(30000, min(calculated_value, 120000000))  # Global min/max limits
            
        except Exception as e:
            print(f"Error in value calculation: {str(e)}")
            return 30000  # Fallback minimum value

    def _categorize_player(self, player):
        """Enhanced player categorization with stricter criteria"""
        overall = player['Overall']
        potential = player['Potential']
        value = player['Value(£)']
        
        if (overall >= self.category_thresholds['STRONG']['min_overall'] or 
            potential >= self.category_thresholds['STRONG']['min_potential'] or 
            value >= self.category_thresholds['STRONG']['min_value']):
            return 'STRONG'
        elif (overall >= self.category_thresholds['MIDDLE']['min_overall'] or 
              potential >= self.category_thresholds['MIDDLE']['min_potential'] or 
              value >= self.category_thresholds['MIDDLE']['min_value']):
            return 'MIDDLE'
        else:
            return 'WEAK'

    def _initialize_model(self):
        """Initialize and train the model with enhanced validation"""
        if self.all_players.empty:
            print("Warning: No training data available. Using default model parameters.")
            return

        try:
            # Create advanced features with proper error handling
            X = self.model.process_player_data(self.all_players)
            
            # Map categories to numeric values
            category_mapping = {'STRONG': 2, 'MIDDLE': 1, 'WEAK': 0}
            
            # Add category information for training with error handling
            X['category_code'] = self.all_players['category'].map(category_mapping).fillna(1)
            
            # Prepare target variables with validation
            y_value = self.all_players['Value(£)'].clip(lower=30000)
            y_category = X['category_code']
            
            # Ensure we have enough training data
            if len(X) < 10:
                print("Warning: Limited training data available. Model predictions may be less accurate.")
            
            # Train the enhanced model
            print("Training model with", len(X), "players...")
            self.model.train(X, y_value, y_category)
            print("Model training completed successfully.")
            
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            print("Using fallback model parameters.")
            
            # Create minimal training data
            minimal_data = pd.DataFrame({
                'Age': [20, 25, 30],
                'Overall': [75, 85, 65],
                'Potential': [80, 90, 70],
                'Value(£)': [500000, 2000000, 100000],
                'speed': [70, 85, 60],
                'dodge': [70, 85, 60],
                'Height(cm.)': [180, 185, 175],
                'Weight(lbs.)': [170, 180, 165],
                'category': ['MIDDLE', 'STRONG', 'WEAK']
            })
            
            X = self.model.process_player_data(minimal_data)
            X['category_code'] = minimal_data['category'].map(category_mapping)
            self.model.train(X, minimal_data['Value(£)'], X['category_code'])

    def analyze_new_player(self):
        """Get and analyze new player data with enhanced categorization"""
        print("\n=== Enhanced Player Analysis ===")
        
        player_data = self._get_player_input()
        if player_data is None:
            return None

        # Validate and process player data
        player_data = self._validate_player_data(player_data)
        
        # Get AI evaluation with enhanced category prediction
        evaluation = self.model.evaluate_player(player_data)
        
        # Find similar players within the same category
        similar_players = self._find_similar_players(player_data, evaluation['category'])
        
        # Generate report using the correct method name
        return self._generate_report(player_data, evaluation, similar_players)

    def _validate_player_data(self, player_data):
        """Enhanced player data validation with higher thresholds"""
        try:
            # Validate numeric ranges with enhanced limits
            player_data['Age'] = abs(player_data['Age'].clip(16, 45))
            player_data['Overall'] = abs(player_data['Overall'].clip(1, 99))  # Allow very high ratings
            player_data['Potential'] = abs(player_data['Potential'].clip(1, 99))
            player_data['International Reputation'] = abs(player_data['International Reputation'].clip(1, 5))
            player_data['speed'] = abs(player_data['speed'].clip(1, 99))
            player_data['dodge'] = abs(player_data['dodge'].clip(1, 99))
            player_data['Height(cm.)'] = abs(player_data['Height(cm.)'].clip(150, 220))
            player_data['Weight(lbs.)'] = abs(player_data['Weight(lbs.)'].clip(100, 250))
            
            # Enhanced potential calculation
            current_overall = player_data['Overall'].iloc[0]
            potential_ceiling = min(99, current_overall + 15)  # Allow high potential but with realistic ceiling
            player_data['Potential'] = player_data['Potential'].clip(
                current_overall, 
                potential_ceiling
            )
            
            return player_data
            
        except Exception as e:
            print(f"Error validating player data: {str(e)}")
            return player_data

    def _get_player_input(self):
        """Get new player data with improved validation"""
        try:
            print("\nEnter Player Details:")
            player_data = {
                'name': self._get_text_input("Full Name: "),
                'Age': self._get_numeric_input("Age", 16, 45),
                'Nationality': self._get_text_input("Nationality: "),
                'Overall': self._get_numeric_input("Overall Rating", 1, 100),
                'Potential': self._get_numeric_input("Potential Rating", 1, 100),
                'Club': self._get_text_input("Current Club: "),
                'Value(£)': 30000,  # Default minimum value
                'Preferred Foot': self._get_foot_input(),
                'International Reputation': self._get_numeric_input("International Reputation", 1, 5),
                'speed': self._get_numeric_input("Speed Rating", 1, 100),
                'dodge': self._get_numeric_input("Agility Rating", 1, 100),
                'Height(cm.)': self._get_numeric_input("Height (cm)", 150, 220),
                'Weight(lbs.)': self._get_numeric_input("Weight (lbs)", 100, 250)
            }
            return pd.DataFrame([player_data])
            
        except ValueError as e:
            print(f"\nError: {str(e)}")
            return None

    def _get_numeric_input(self, prompt, min_val, max_val):
        """Get and validate numeric input"""
        while True:
            try:
                value = float(input(f"{prompt} ({min_val}-{max_val}): "))
                if min_val <= value <= max_val:
                    return value
                print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")

    def _get_text_input(self, prompt):
        """Get and validate text input"""
        while True:
            value = input(prompt).strip()
            if value:
                return value
            print("This field cannot be empty")

    def _get_foot_input(self):
        """Get and validate preferred foot input"""
        while True:
            foot = input("Preferred Foot (Right/Left): ").capitalize()
            if foot in ['Right', 'Left']:
                return foot
            print("Please enter 'Right' or 'Left'")

    def _find_similar_players(self, new_player, category, num_similar=5):
        """Find similar players with enhanced category matching"""
        try:
            # Filter players by category
            category_players = self.all_players[self.all_players['category'] == category]
            
            if category_players.empty:
                return []
            
            # Calculate similarity scores
            features = ['Age', 'Overall', 'Potential', 'speed', 'dodge']
            weights = {'Age': 0.15, 'Overall': 0.3, 'Potential': 0.3, 'speed': 0.15, 'dodge': 0.1}
            
            similarities = []
            new_values = new_player[features].iloc[0]
            
            for _, player in category_players.iterrows():
                weighted_diff = sum(
                    weights[feature] * abs(new_values[feature] - player[feature])
                    for feature in features
                )
                similarities.append((player, weighted_diff))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1])
            return [player for player, _ in similarities[:num_similar]]
            
        except Exception as e:
            print(f"Error finding similar players: {str(e)}")
            return []

    def _generate_report(self, player_data, evaluation, similar_players):
        """Generate detailed analysis report"""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("FOOTBALL TALENT SCOUT - PLAYER ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Basic Information
        report.append("\nPLAYER INFORMATION")
        report.append("-" * 20)
        report.append(f"Name: {player_data['name'].iloc[0]}")
        report.append(f"Age: {player_data['Age'].iloc[0]}")
        report.append(f"Nationality: {player_data['Nationality'].iloc[0]}")
        report.append(f"Current Club: {player_data['Club'].iloc[0]}")
        report.append(f"Preferred Foot: {player_data['Preferred Foot'].iloc[0]}")
        
        # Performance Metrics
        report.append("\nPERFORMANCE METRICS")
        report.append("-" * 20)
        report.append(f"Overall Rating: {player_data['Overall'].iloc[0]}")
        report.append(f"Potential Rating: {player_data['Potential'].iloc[0]}")
        report.append(f"Speed Rating: {player_data['speed'].iloc[0]}")
        report.append(f"Agility Rating: {player_data['dodge'].iloc[0]}")
        
        # AI Evaluation
        report.append("\nAI EVALUATION RESULTS")
        report.append("-" * 20)
        report.append(f"Development Level: {evaluation['rating']}")
        report.append(f"Potential Score: {evaluation['potential_score']:.2f}")
        report.append(f"Confidence Level: {evaluation['confidence']:.2%}")
        report.append(f"Estimated Market Value: £{evaluation['market_value']:,.2f}")
        
        # Strengths and Development
        report.append("\nKEY STRENGTHS")
        report.append("-" * 20)
        for strength in evaluation['strengths']:
            report.append(f"• {strength}")
        
        report.append("\nDEVELOPMENT AREAS")
        report.append("-" * 20)
        for need in evaluation['development_needs']:
            report.append(f"• {need}")
        
        # Similar Players Comparison
        report.append("\nSIMILAR PLAYERS COMPARISON")
        report.append("-" * 20)
        report.append(f"{'Name':<25} {'Age':>3} {'Overall':>8} {'Potential':>9} {'Value(£)':>12}")
        report.append("-" * 65)
        
        # New player's data
        report.append(f"{player_data['name'].iloc[0]:<25} {player_data['Age'].iloc[0]:>3.0f} "
                     f"{player_data['Overall'].iloc[0]:>8.0f} {player_data['Potential'].iloc[0]:>9.0f} "
                     f"{evaluation['market_value']:>12,.0f}")
        
        # Similar players' data
        for player in similar_players:
            report.append(f"{player['name']:<25} {player['Age']:>3.0f} {player['Overall']:>8.0f} "
                         f"{player['Potential']:>9.0f} {player['Value(£)']:>12,.0f}")
        
        # Final Recommendation
        report.append("\nRECOMMENDATION")
        report.append("-" * 20)
        report.append(evaluation['recommendation'])
        
        # Save report
        report_text = "\n".join(report)
        filename = f"reports/{player_data['name'].iloc[0].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
        
        os.makedirs('reports', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text, filename

def main():
    scout = FootballScout()
    
    while True:
        print("\n=== Football Talent Scout AI System ===")
        print("1. Analyze New Player")
        print("2. Exit")
        
        choice = input("\nSelect an option (1-2): ")
        
        if choice == "1":
            result = scout.analyze_new_player()
            if result:
                report, filename = result
                print(f"\nAnalysis complete! Report saved to: {filename}")
                print("\nReport Preview:")
                print("=" * 60)
                print(report)
                
        elif choice == "2":
            print("\nThank you for using Football Talent Scout AI!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()