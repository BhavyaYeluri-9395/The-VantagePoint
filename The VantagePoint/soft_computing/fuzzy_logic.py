import numpy as np
import json
import os

class FuzzyLogicSystem:
    """
    Enhanced Fuzzy Logic System with Knowledge Base and Complete Defuzzifier
    """
    
    def __init__(self, knowledge_base_path="knowledge_base/fuzzy_rules.json"):
        self.knowledge_base_path = knowledge_base_path
        self.rules = []
        self.input_mfs = {}
        self.output_mfs = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load fuzzy rules and membership functions from JSON knowledge base"""
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'r') as f:
                kb = json.load(f)
            
            self.input_mfs = kb.get("input_variables", {})
            self.output_mfs = kb.get("output_variables", {})
            self.rules = kb.get("rules", [])
            print(f"Loaded {len(self.rules)} fuzzy rules from knowledge base")
        else:
            print("Knowledge base not found, using default rules")
            self._initialize_default_kb()
    
    def _initialize_default_kb(self):
        """Initialize default knowledge base if JSON not found"""
        self.input_mfs = {
            "budget": {
                "low": {"params": [0, 0, 150000000]},
                "medium": {"params": [80000000, 150000000, 250000000]},
                "high": {"params": [200000000, 300000000, 500000000]}
            },
            "popularity": {
                "weak": {"params": [0, 0, 40]},
                "moderate": {"params": [30, 60, 80]},
                "strong": {"params": [70, 90, 100]}
            }
        }
        self.output_mfs = {
            "fuzzy_score": {
                "poor": {"params": [0, 0, 4]},
                "average": {"params": [3, 5, 7]},
                "good": {"params": [6, 8, 10]}
            }
        }
        self.rules = [
            {"if": {"budget": "high", "popularity": "strong"}, "then": {"fuzzy_score": "good"}, "weight": 9.5},
            {"if": {"budget": "high", "popularity": "moderate"}, "then": {"fuzzy_score": "good"}, "weight": 7.5},
            {"if": {"budget": "medium", "popularity": "moderate"}, "then": {"fuzzy_score": "average"}, "weight": 6.5},
            {"if": {"budget": "medium", "popularity": "strong"}, "then": {"fuzzy_score": "good"}, "weight": 8.0},
            {"if": {"budget": "low", "popularity": "weak"}, "then": {"fuzzy_score": "poor"}, "weight": 3.5},
            {"if": {"budget": "low", "popularity": "moderate"}, "then": {"fuzzy_score": "average"}, "weight": 5.0}
        ]
    
    def save_knowledge_base(self):
        """Save current fuzzy rules to knowledge base"""
        kb = {
            "input_variables": self.input_mfs,
            "output_variables": self.output_mfs,
            "rules": self.rules
        }
        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
        with open(self.knowledge_base_path, 'w') as f:
            json.dump(kb, f, indent=2)
        print("Knowledge base saved")
    
    def triangular_mf(self, x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a + 1e-9)
        else:
            return (c - x) / (c - b + 1e-9)
    
    def fuzzify(self, value, var_name, mf_name):
        """Fuzzify a crisp value using membership functions from knowledge base"""
        if var_name in self.input_mfs and mf_name in self.input_mfs[var_name]:
            params = self.input_mfs[var_name][mf_name]["params"]
            return self.triangular_mf(value, params[0], params[1], params[2])
        return 0.0
    
    def evaluate_rules(self, budget, popularity):
        """Evaluate fuzzy rules from knowledge base"""
        rule_outputs = []
        
        for rule in self.rules:
            # Calculate firing strength (min of antecedents)
            firing_strength = 1.0
            for var, mf in rule["if"].items():
                if var == "budget":
                    membership = self.fuzzify(budget, "budget", mf)
                elif var == "popularity":
                    membership = self.fuzzify(popularity, "popularity", mf)
                else:
                    membership = 0
                firing_strength = min(firing_strength, membership)
            
            # Get rule weight and conclusion
            weight = rule.get("weight", 5.0)
            conclusion = rule["then"]["fuzzy_score"]
            
            # Get output membership function parameters
            if conclusion in self.output_mfs["fuzzy_score"]:
                params = self.output_mfs["fuzzy_score"][conclusion]["params"]
                rule_outputs.append({
                    "firing_strength": firing_strength,
                    "weight": weight,
                    "output_params": params
                })
        
        return rule_outputs
    
    def defuzzify(self, rule_outputs):
        """
        Complete Defuzzifier using Centroid Method (Center of Gravity)
        This is the most accurate defuzzification method
        """
        if not rule_outputs:
            return 5.0  # Default middle value
        
        # Sample the output space for centroid calculation
        sample_points = np.linspace(0, 10, 1000)
        aggregated_membership = np.zeros_like(sample_points)
        
        for rule in rule_outputs:
            firing_strength = rule["firing_strength"] * (rule["weight"] / 10.0)
            params = rule["output_params"]
            
            for i, x in enumerate(sample_points):
                mf_value = self.triangular_mf(x, params[0], params[1], params[2])
                aggregated_membership[i] = max(aggregated_membership[i], firing_strength * mf_value)
        
        # Centroid calculation: ∫ x * μ(x) dx / ∫ μ(x) dx
        numerator = np.sum(sample_points * aggregated_membership)
        denominator = np.sum(aggregated_membership)
        
        if denominator > 0:
            centroid = numerator / denominator
        else:
            centroid = 5.0
        
        return max(0, min(10, centroid))
    
    def compute_fuzzy_score(self, budget, popularity, use_knowledge_base=True):
        """
        Compute fuzzy score using either knowledge base or direct rules
        """
        if use_knowledge_base and self.rules:
            # Use knowledge base rules
            rule_outputs = self.evaluate_rules(budget, popularity)
            fuzzy_score = self.defuzzify(rule_outputs)
        else:
            # Fallback to direct calculation (backward compatibility)
            fuzzy_score = self._legacy_compute_fuzzy_score(budget, popularity)
        
        return fuzzy_score
    
    def _legacy_compute_fuzzy_score(self, budget, popularity):
        """Legacy method for backward compatibility"""
        # Normalize values
        budget_norm = min(1.0, budget / 500_000_000)
        popularity_norm = min(1.0, popularity / 100)
        
        # Membership functions
        low_budget = self.triangular_mf(budget_norm, 0, 0, 0.3)
        medium_budget = self.triangular_mf(budget_norm, 0.2, 0.5, 0.8)
        high_budget = self.triangular_mf(budget_norm, 0.6, 0.8, 1)
        
        low_pop = self.triangular_mf(popularity_norm, 0, 0, 0.3)
        medium_pop = self.triangular_mf(popularity_norm, 0.2, 0.5, 0.8)
        high_pop = self.triangular_mf(popularity_norm, 0.6, 0.8, 1)
        
        # Fuzzy rules
        rule_outputs = [
            {"firing_strength": min(high_budget, high_pop), "value": 9.5},
            {"firing_strength": min(medium_budget, medium_pop), "value": 6.5},
            {"firing_strength": min(low_budget, low_pop), "value": 3.5},
            {"firing_strength": min(high_budget, medium_pop), "value": 7.5},
            {"firing_strength": min(medium_budget, high_pop), "value": 8.0}
        ]
        
        numerator = sum(r["firing_strength"] * r["value"] for r in rule_outputs)
        denominator = sum(r["firing_strength"] for r in rule_outputs)
        
        return numerator / denominator if denominator > 0 else 5.0
    
    def update_membership_function(self, var_name, mf_name, param_index, new_value):
        """Update membership function parameters (for PSO optimization)"""
        if var_name in self.input_mfs and mf_name in self.input_mfs[var_name]:
            old_params = self.input_mfs[var_name][mf_name]["params"]
            old_params[param_index] = new_value
            self.input_mfs[var_name][mf_name]["params"] = old_params
            return True
        return False
    
    def get_membership_parameters(self):
        """Get all membership function parameters for optimization"""
        params = []
        param_names = []
        
        for var_name, mfs in self.input_mfs.items():
            for mf_name, mf_data in mfs.items():
                for i, param in enumerate(mf_data["params"]):
                    params.append(param)
                    param_names.append(f"{var_name}_{mf_name}_p{i}")
        
        return np.array(params), param_names
    
    def set_membership_parameters(self, params, param_names):
        """Set membership function parameters from optimization"""
        param_dict = dict(zip(param_names, params))
        
        for var_name, mfs in self.input_mfs.items():
            for mf_name in mfs:
                for i in range(3):
                    key = f"{var_name}_{mf_name}_p{i}"
                    if key in param_dict:
                        self.input_mfs[var_name][mf_name]["params"][i] = param_dict[key]