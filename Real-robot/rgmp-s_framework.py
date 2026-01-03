import numpy as np
import torch
import torch.nn as nn
import cv2
import time
from gaussian_mixture_model import GMMActionOptimizer
from RASNet import *
from lgss_framework import LGSS_Framework

class RGMPSFramework:
    """
    Recurrent Geometric-prior Multimodal Policy-S (RGMP-S) Framework
    
    Consists of two key components:
    1. LGSS (Language-Guided Skill Selection) - translates verbal commands into executable skill sequences
    2. RASNet (Recursive Adaptive Spiking Network) - processes RGB images using RASNet architecture to predict joint angles
    
    The framework uses RASNet architecture with DenseNet blocks and adaptive spiking neurons for enhanced spatial attention and geometric understanding.
    """
    
    def __init__(self, model_path='models/rasnet_model.pth', gmm_path='models/gmm_model.pkl'):
        """
        Initialize RGMP-S Framework
        
        Args:
            model_path: Path to trained RASNet model
            gmm_path: Path to trained GMM model
        """
        # Initialize LGSS (Language-Guided Skill Selection)
        self.lgss = LGSS_Framework()
        
        # Initialize RASNet (Recursive Adaptive Spiking Network)
        self.rasnet = self._load_rasnet_model(model_path)
        
        # Initialize GMM Action Optimizer
        self.gmm_optimizer = GMMActionOptimizer(gmm_path)
        
        # Demonstration data storage
        self.demonstration_data = []
        
        print("RGMP-S Framework initialized successfully")
    
    def _load_rasnet_model(self, model_path):
        """Load trained RASNet model with RASNet architecture"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            model = RASNet_ImageModel(num_classes=6)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            print(f"RASNet model with RASNet loaded from {model_path}")
            return model
            
        except Exception as e:
            print(f"Failed to load RASNet model: {e}")
            # Return default RASNet model
            model = RASNet_ImageModel(num_classes=6)
            model.eval()
            return model
    
    def collect_demonstration(self, observation, joint_angles):
        """
        Collect human demonstration data
        
        Args:
            observation: RGB image observation (H, W, 3)
            joint_angles: Ground truth joint angles (6,)
        
        Returns:
            demonstration_id: ID of stored demonstration
        """
        demonstration = {
            'observation': observation,
            'joint_angles': np.array(joint_angles),
            'timestamp': time.time()
        }
        
        self.demonstration_data.append(demonstration)
        demonstration_id = len(self.demonstration_data) - 1
        
        print(f"Collected demonstration {demonstration_id}: joints = {joint_angles}")
        
        return demonstration_id
    
    def train_gmm(self, epochs=100):
        """
        Train GMM using collected demonstration data
        
        Args:
            epochs: Number of training epochs (for compatibility, GMM uses EM algorithm)
        
        Returns:
            trained_gmm: Trained GMM model
        """
        if len(self.demonstration_data) < 6:
            print("Warning: Need at least 6 demonstrations for GMM training")
            return None
        
        # Extract joint angles from demonstrations
        joint_configurations = []
        for demo in self.demonstration_data:
            joint_configurations.append(demo['joint_angles'])
        
        joint_configurations = np.array(joint_configurations)
        
        print(f"Training GMM with {len(joint_configurations)} demonstrations")
        
        # Train GMM using EM algorithm
        self.gmm_optimizer.train_gmm(joint_configurations)
        
        # Save trained GMM
        self.gmm_optimizer.save_gmm('models/gmm_model.pkl')
        
        print("GMM training completed")
        return self.gmm_optimizer.gmm
    
    def inference_pipeline(self, instruction, observation):
        """
        Complete RGMP-S inference pipeline
        
        Args:
            instruction: Human speech instruction
            observation: RGB image observation (H, W, 3)
        
        Returns:
            final_action: Optimized joint angles (6,)
            pipeline_info: Detailed pipeline information
        """
        pipeline_start = time.time()
        
        # Stage 1: LGSS - Language-Guided Skill Selection
        print(f"Processing instruction: {instruction}")
        
        # Use LGSS to get bounding box and skill selection
        lgss_result = self.lgss.process_instruction(observation, instruction)
        
        bbox = lgss_result['bounding_box']
        selected_skill = lgss_result['selected_skill']
        shape_info = lgss_result['shape_info']
        
        print(f"LGSS Result - Skill: {selected_skill}, Shape: {shape_info}, BBox: {bbox}")
        
        # Stage 2: RASNet - Process RGB image for joint angle prediction
        initial_action = self._rasnet_inference(observation)
        
        print(f"RASNet Initial Prediction: {initial_action}")
        
        # Stage 3: GMM Optimization
        if self.gmm_optimizer.gmm.is_fitted:
            final_action = self.gmm_optimizer.optimize_action(initial_action)
            print(f"GMM Optimized Action: {final_action}")
        else:
            final_action = initial_action
            print("GMM not available, using RASNet prediction")
        
        pipeline_time = (time.time() - pipeline_start) * 1000
        
        pipeline_info = {
            'instruction': instruction,
            'lgss_result': lgss_result,
            'initial_action': initial_action,
            'final_action': final_action,
            'processing_time_ms': pipeline_time,
            'selected_skill': selected_skill,
            'shape_info': shape_info
        }
        
        print(f"RGMP-S Pipeline completed in {pipeline_time:.1f}ms")
        
        return final_action, pipeline_info
    
    def _rasnet_inference(self, observation):
        """
        RASNet inference for joint angle prediction
        
        Args:
            observation: RGB image (H, W, 3)
        
        Returns:
            predicted_joints: Predicted joint angles (6,)
        """
        # Preprocess image for RASNet
        if isinstance(observation, str):
            # If observation is file path
            image = cv2.imread(observation)
        else:
            image = observation
        
        # Resize and normalize for RASNet architecture
        image = cv2.resize(image, (224, 224))  # RASNet uses 224x224
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        
        # RASNet inference
        device = next(self.rasnet.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            predicted_joints = self.rasnet(image_tensor)
            predicted_joints = predicted_joints.cpu().numpy()[0]  # Remove batch dimension
        
        return predicted_joints
    
    def update_gmm_parameters(self, new_demonstration):
        """
        Update GMM parameters with new demonstration (online learning)
        
        Args:
            new_demonstration: New demonstration data (observation, joint_angles)
        """
        observation, joint_angles = new_demonstration
        
        # Add to demonstration data
        demo_id = self.collect_demonstration(observation, joint_angles)
        
        # Retrain GMM with updated data
        if len(self.demonstration_data) >= 6:
            self.train_gmm()
            print(f"GMM updated with demonstration {demo_id}")
        
        return demo_id
    
    def get_gmm_cluster_analysis(self):
        """
        Get detailed GMM cluster analysis
        
        Returns:
            cluster_analysis: Dictionary with cluster information
        """
        if not self.gmm_optimizer.gmm.is_fitted:
            return {"error": "GMM not fitted"}
        
        cluster_info = self.gmm_optimizer.gmm.get_cluster_info()
        
        # Add additional analysis
        cluster_analysis = {
            'basic_info': cluster_info,
            'n_demonstrations': len(self.demonstration_data),
            'cluster_centers': cluster_info['means'],
            'cluster_weights': cluster_info['priors'],
            'convergence_status': cluster_info['converged']
        }
        
        return cluster_analysis
    
    def save_framework_state(self, save_dir='models/'):
        """
        Save complete framework state
        
        Args:
            save_dir: Directory to save framework components
        """
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save GMM model
        self.gmm_optimizer.save_gmm(os.path.join(save_dir, 'gmm_model.pkl'))
        
        # Save demonstration data
        demo_file = os.path.join(save_dir, 'demonstration_data.pkl')
        with open(demo_file, 'wb') as f:
            pickle.dump(self.demonstration_data, f)
        
        # Save framework configuration
        config = {
            'n_demonstrations': len(self.demonstration_data),
            'gmm_fitted': self.gmm_optimizer.gmm.is_fitted,
            'save_timestamp': time.time()
        }
        
        config_file = os.path.join(save_dir, 'rgmp-s_config.pkl')
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"RGMP-S Framework state saved to {save_dir}")
    
    def load_framework_state(self, save_dir='models/'):
        """
        Load complete framework state
        
        Args:
            save_dir: Directory containing saved framework components
        """
        import os
        import pickle
        
        # Load GMM model
        gmm_file = os.path.join(save_dir, 'gmm_model.pkl')
        if os.path.exists(gmm_file):
            self.gmm_optimizer.load_gmm(gmm_file)
        
        # Load demonstration data
        demo_file = os.path.join(save_dir, 'demonstration_data.pkl')
        if os.path.exists(demo_file):
            with open(demo_file, 'rb') as f:
                self.demonstration_data = pickle.load(f)
        
        # Load framework configuration
        config_file = os.path.join(save_dir, 'rgmp-s_config.pkl')
        if os.path.exists(config_file):
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
                print(f"Loaded RGMP-S config: {config}")
        
        print(f"RGMP-S Framework state loaded from {save_dir}")

# Algorithm implementation following the pseudocode
class RGMPSAlgorithm:
    """
    Implementation of Algorithm 1: The RGMP-S Framework
    """
    
    def __init__(self):
        self.rgmps_framework = RGMPSFramework()
        self.conversation_history = []
    
    def human_demonstration_collecting(self, M=100):
        """
        Human Demonstration Collecting phase
        
        Args:
            M: Capacity of demonstration dataset
        
        Returns:
            demonstration_dataset: Collected demonstrations
        """
        print("=== Human Demonstration Collecting Phase ===")
        
        demonstration_dataset = []
        
        for i in range(1, M + 1):
            print(f"Collecting demonstration {i}/{M}")
            
            # Simulate demonstration collection
            # In real implementation, this would capture actual human demonstrations
            observation = self._simulate_observation()
            joint_angles = self._simulate_joint_angles()
            
            # Store demonstration
            demo_data = {
                'observation': observation,
                'joint_angles': joint_angles,
                'demo_id': i
            }
            
            demonstration_dataset.append(demo_data)
            self.rgmps_framework.collect_demonstration(observation, joint_angles)
        
        print(f"Collected {len(demonstration_dataset)} demonstrations")
        return demonstration_dataset
    
    def rgmps_training_pipeline(self, demonstration_dataset, E=100):
        """
        RGMP-S Training Pipeline
        
        Args:
            demonstration_dataset: Human demonstrations
            E: Training epochs
        
        Returns:
            trained_model: Trained RGMP-S model
            gmm_parameters: Trained GMM parameters
        """
        print("=== RGMP-S Training Pipeline ===")
        
        for epoch in range(1, E + 1):
            print(f"Training epoch {epoch}/{E}")
            
            # Update RASNet parameters (simulated)
            # In real implementation, this would train the RASNet network
            self._simulate_rasnet_training(demonstration_dataset)
            
            # Update GMM parameters
            if epoch % 10 == 0:  # Update GMM every 10 epochs
                self.rgmps_framework.train_gmm()
        
        print("RGMP-S training completed")
        
        return self.rgmps_framework.rasnet, self.rgmps_framework.gmm_optimizer.gmm
    
    def inference_pipeline(self, T=10):
        """
        Inference Pipeline for T conversation rounds
        
        Args:
            T: Number of conversation rounds
        
        Returns:
            conversation_results: Results from all conversation rounds
        """
        print("=== Inference Pipeline ===")
        
        conversation_results = []
        
        for t in range(1, T + 1):
            print(f"Conversation round {t}/{T}")
            
            # Simulate human speech and observation
            instruction = self._simulate_human_instruction()
            observation = self._simulate_observation()
            
            # Process through RGMP-S pipeline
            final_action, pipeline_info = self.rgmps_framework.inference_pipeline(
                instruction, observation
            )
            
            # Store conversation result
            result = {
                'round': t,
                'instruction': instruction,
                'final_action': final_action,
                'pipeline_info': pipeline_info
            }
            
            conversation_results.append(result)
            self.conversation_history.append(result)
            
            print(f"Round {t} completed - Action: {final_action}")
        
        return conversation_results
    
    def _simulate_observation(self):
        """Simulate RGB observation"""
        # Generate random RGB image for simulation
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def _simulate_joint_angles(self):
        """Simulate ground truth joint angles"""
        # Generate realistic joint angles for 6-DOF arm
        return np.random.uniform(-3.14, 3.14, 6)
    
    def _simulate_human_instruction(self):
        """Simulate human speech instruction"""
        instructions = [
            "I want Fanta",
            "Pick up the can",
            "Grasp the bottle",
            "Lift the object",
            "Get the drink",
            "Take the container"
        ]
        return np.random.choice(instructions)
    
    def _simulate_rasnet_training(self, demonstration_dataset):
        """Simulate RASNet training process"""
        # In real implementation, this would train the RASNet network
        # using the demonstration dataset
        pass

# Global RGMP-S framework instance
rgmps_framework = None

def initialize_rgmps_framework():
    """Initialize global RGMP-S framework"""
    global rgmps_framework
    if rgmps_framework is None:
        rgmps_framework = RGMPSFramework()
    return rgmps_framework

def process_rgmps_instruction(instruction, observation):
    """Process instruction through RGMP-S framework"""
    if rgmps_framework is None:
        initialize_rgmps_framework()
    return rgmps_framework.inference_pipeline(instruction, observation)

def add_rgmps_demonstration(observation, joint_angles):
    """Add demonstration to RGMP-S framework"""
    if rgmps_framework is None:
        initialize_rgmps_framework()
    return rgmps_framework.collect_demonstration(observation, joint_angles)

if __name__ == "__main__":
    # Test RGMP-S Algorithm implementation
    print("Testing RGMP-S Algorithm Implementation")
    
    # Initialize algorithm
    algorithm = RGMPSAlgorithm()
    
    # Phase 1: Human Demonstration Collecting
    demonstrations = algorithm.human_demonstration_collecting(M=50)
    
    # Phase 2: RGMP-S Training Pipeline
    trained_model, gmm_params = algorithm.rgmps_training_pipeline(demonstrations, E=20)
    
    # Phase 3: Inference Pipeline
    results = algorithm.inference_pipeline(T=5)
    
    # Print summary
    print("\n=== RGMP-S Algorithm Summary ===")
    print(f"Demonstrations collected: {len(demonstrations)}")
    print(f"GMM clusters: {algorithm.rgmps_framework.gmm_optimizer.gmm.K}")
    print(f"Conversation rounds: {len(results)}")
    
    # Analyze GMM clusters
    cluster_analysis = algorithm.rgmps_framework.get_gmm_cluster_analysis()
    print(f"GMM convergence: {cluster_analysis['convergence_status']}")
    print(f"Cluster weights: {cluster_analysis['cluster_weights']}")
    
    # Save framework state
    algorithm.rgmps_framework.save_framework_state()
    
    print("RGMP-S Algorithm test completed successfully!")