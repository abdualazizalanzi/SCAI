import pandas as pd         
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

class TalentScoutAI:
    def __init__(self):
        self.value_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.category_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_fitted = False
        self.training_features = [
            'Age', 'Overall', 'Potential', 'speed', 'dodge',
            'potential_growth', 'value_per_rating', 'speed_to_age_ratio',
            'agility_score', 'physical_index', 'performance_score',
            'weighted_potential'
        ]

    def process_player_data(self, data):
        """Process player data with enhanced feature engineering"""
        processed_data = data.copy()
        
        # Advanced feature engineering
        processed_data['potential_growth'] = processed_data['Potential'] - processed_data['Overall']
        processed_data['value_per_rating'] = processed_data['Value(£)'] / processed_data['Overall']
        processed_data['speed_to_age_ratio'] = processed_data['speed'] / processed_data['Age']
        processed_data['agility_score'] = processed_data['dodge'] * 1.2  # Increased importance
        
        # Physical attributes impact
        processed_data['physical_index'] = (
            (processed_data['Height(cm.)'] / 180) * 0.6 +
            (processed_data['Weight(lbs.)'] / 170) * 0.4
        )
        
        # Create performance score
        processed_data['performance_score'] = (
            processed_data['Overall'] * 0.4 +
            processed_data['Potential'] * 0.3 +
            processed_data['speed'] * 0.15 +
            processed_data['dodge'] * 0.15
        )
        
        # Advanced potential calculation
        age_factor = np.where(
            processed_data['Age'] <= 23,
            1.2,  # Younger players have higher potential
            np.where(
                processed_data['Age'] <= 28,
                1.0,  # Prime age
                0.8   # Older players
            )
        )
        processed_data['weighted_potential'] = processed_data['Potential'] * age_factor
        
        return processed_data

    def train(self, X, y_value, y_category):
        """Train the model with enhanced validation and feature importance analysis"""
        processed_data = self.process_player_data(X)
        
        # Prepare features for training
        X_train = processed_data[self.training_features]
        
        # Fit and transform the scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train value prediction model
        self.value_model.fit(X_train_scaled, y_value)
        
        # Train category classification model
        self.category_model.fit(X_train_scaled, y_category)
        
        # Calculate and store feature importance
        self.feature_importance = dict(zip(
            self.training_features,
            self.value_model.feature_importances_
        ))
        
        self.is_fitted = True

    def evaluate_player(self, player_data):
        """Enhanced player evaluation with sophisticated analysis"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Please call train() first.")
            
        processed_data = self.process_player_data(player_data)
        
        # Extract features in the same order as training
        X = processed_data[self.training_features]
        
        # Transform using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get base prediction from model
        base_predicted_value = float(self.value_model.predict(X_scaled)[0])
        
        # Apply enhanced value calculation logic
        overall = player_data['Overall'].iloc[0]
        potential = player_data['Potential'].iloc[0]
        age = player_data['Age'].iloc[0]
        
        # Base value calculation based on overall rating
        base_value = overall * 1000
        
        # Potential bonus
        potential_bonus = (potential - overall) * 500
        
        # Age factor - younger players get bonus
        age_factor = max(0.5, (30 - age) / 10) if age <= 30 else 0.3
        
        # Calculate final value
        calculated_value = (base_value + potential_bonus) * age_factor
        
        # Apply thresholds based on player level
        if overall >= 80 or potential >= 85:  # Strong players
            calculated_value = max(calculated_value, 1000000)
        elif overall >= 70 or potential >= 75:  # Middle tier
            calculated_value = min(max(calculated_value, 500000), 999999)
        else:  # Weak players
            calculated_value = min(calculated_value, 499999)
            # Ensure very low-rated players get very low values
            if overall < 30:
                calculated_value = min(calculated_value, 50000)
        
        # Use the minimum of model prediction and calculated value for consistency
        predicted_value = min(base_predicted_value, calculated_value)
        predicted_value = max(30000, min(predicted_value, 120000000))  # Global limits
        
        # Rest of the evaluation
        category_proba = self.category_model.predict_proba(X_scaled)[0]
        predicted_category = self.category_model.predict(X_scaled)[0]
        
        # Map category code back to label
        category_map = {0: 'WEAK', 1: 'MIDDLE', 2: 'STRONG'}
        category = category_map[predicted_category]
        
        # Calculate confidence level
        confidence = float(max(category_proba))
        
        # Enhanced potential score calculation
        potential_score = (
            processed_data['weighted_potential'].iloc[0] * 0.4 +
            processed_data['performance_score'].iloc[0] * 0.3 +
            processed_data['physical_index'].iloc[0] * 0.3
        )
        
        # Determine player strengths and development needs
        strengths = self._analyze_strengths(processed_data)
        development_needs = self._analyze_development_needs(processed_data)
        
        # Generate detailed recommendation
        recommendation = self._generate_recommendation(
            processed_data,
            category,
            potential_score,
            confidence
        )
        
        return {
            'market_value': predicted_value,
            'category': category,
            'confidence': confidence,
            'potential_score': potential_score,
            'strengths': strengths,
            'development_needs': development_needs,
            'recommendation': recommendation,
            'rating': self._get_rating(potential_score)
        }

    def _analyze_strengths(self, data):
        """Identify player strengths with enhanced criteria"""
        strengths = []
        
        # Physical attributes
        if data['speed'].iloc[0] >= 85:
            strengths.append("Exceptional Speed")
        elif data['speed'].iloc[0] >= 75:
            strengths.append("Good Speed")
            
        if data['dodge'].iloc[0] >= 85:
            strengths.append("Excellent Agility")
        elif data['dodge'].iloc[0] >= 75:
            strengths.append("Good Agility")
            
        # Overall performance
        if data['Overall'].iloc[0] >= 85:
            strengths.append("Elite Technical Skills")
        elif data['Overall'].iloc[0] >= 75:
            strengths.append("Strong Technical Foundation")
            
        # Potential
        if data['Potential'].iloc[0] >= 90:
            strengths.append("World-Class Potential")
        elif data['Potential'].iloc[0] >= 80:
            strengths.append("High Development Potential")
            
        # Physical build
        if 175 <= data['Height(cm.)'].iloc[0] <= 190:
            strengths.append("Ideal Physical Build")
            
        return strengths

    def _analyze_development_needs(self, data):
        """Identify areas for improvement with enhanced analysis"""
        needs = []
        
        # Physical development
        if data['speed'].iloc[0] < 70:
            needs.append("Speed Enhancement Training Required")
        if data['dodge'].iloc[0] < 70:
            needs.append("Agility Development Needed")
            
        # Technical skills
        if data['Overall'].iloc[0] < 70:
            needs.append("Technical Skills Enhancement Required")
            
        # Potential vs Current
        if (data['Potential'].iloc[0] - data['Overall'].iloc[0]) > 10:
            needs.append("Focus on Realizing Full Potential")
            
        return needs

    def _generate_recommendation(self, data, category, potential_score, confidence):
        """Generate detailed recommendation with enhanced analysis"""
        age = data['Age'].iloc[0]
        overall = data['Overall'].iloc[0]
        potential = data['Potential'].iloc[0]
        
        recommendation = []
        
        # Category-based recommendation
        if category == 'STRONG':
            recommendation.append("Elite talent with exceptional abilities.")
        elif category == 'MIDDLE':
            recommendation.append("Solid performer with good potential.")
        else:
            recommendation.append("Development player with areas for improvement.")
            
        # Age-based recommendation
        if age < 23:
            if potential >= 85:
                recommendation.append("Young prospect with outstanding potential. Priority development recommended.")
            else:
                recommendation.append("Young player with room for development.")
        elif age < 28:
            recommendation.append("Player in prime years. Maximize current performance.")
        else:
            recommendation.append("Experienced player. Focus on maintaining performance level.")
            
        # Confidence-based advice
        if confidence < 0.7:
            recommendation.append("Regular performance monitoring advised.")
            
        return " ".join(recommendation)

    def _get_rating(self, potential_score):
        """Enhanced rating system with more granular categories"""
        if potential_score >= 90:
            return "World Class"
        elif potential_score >= 85:
            return "Elite"
        elif potential_score >= 80:
            return "Very Strong"
        elif potential_score >= 75:
            return "Strong"
        elif potential_score >= 70:
            return "Above Average"
        elif potential_score >= 65:
            return "Average"
        else:
            return "Development Required"