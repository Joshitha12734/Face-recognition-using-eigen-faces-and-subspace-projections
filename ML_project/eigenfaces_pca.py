# eigenfaces_pca.py - Complete PCA from scratch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from data_loader import load_yale_dataset

class EigenfacePCA:
    """
    PCA implementation from scratch for face recognition
    Uses the eigenvalue trick for computational efficiency
    """
    
    def __init__(self, n_components):
        """
        Args:
            n_components: Number of eigenfaces to keep
        """
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        """
        Fit PCA model using the eigenvalue trick
        
        Args:
            X: Training data of shape (n_samples, n_features)
               (n_images, n_pixels) - 165 images × 77760 pixels
        """
        n_samples, n_features = X.shape
        print(f"\nFitting PCA with {n_samples} samples, {n_features} features")
        print(f"Keeping {self.n_components} components")
        
        # Step 1: Compute mean face
        self.mean_face = np.mean(X, axis=0)  # Shape: (n_features,)
        
        # Step 2: Center the data
        X_centered = X - self.mean_face  # Shape: (n_samples, n_features)
        
        # Step 3: The eigenvalue trick!
        print("Computing L matrix (using eigenvalue trick)...")
        L = X_centered @ X_centered.T  # Shape: (n_samples, n_samples)
        
        # Step 4: Eigen decomposition of L
        print("Computing eigenvalues and eigenvectors...")
        eigenvals, eigenvecs_L = np.linalg.eigh(L)
        
        # Step 5: Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs_L = eigenvecs_L[:, idx]
        
        # Step 6: Convert to eigenfaces
        # U = X_centered.T @ eigenvecs_L
        # Then normalize
        print("Computing eigenfaces...")
        self.eigenfaces = X_centered.T @ eigenvecs_L  # Shape: (n_features, n_samples)
        
        # Normalize eigenfaces to unit length
        for i in range(self.eigenfaces.shape[1]):
            norm = np.linalg.norm(self.eigenfaces[:, i])
            if norm > 0:
                self.eigenfaces[:, i] = self.eigenfaces[:, i] / norm
        
        self.eigenvalues = eigenvals
        
        # Step 7: Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues[:self.n_components] / total_variance
        
        # Step 8: Keep only top n_components
        self.eigenfaces = self.eigenfaces[:, :self.n_components]
        self.eigenvalues = self.eigenvalues[:self.n_components]
        
        print(f"✅ PCA fitting complete!")
        print(f"   Variance explained by {self.n_components} components: {np.sum(self.explained_variance_ratio)*100:.2f}%")
        
        return self
    
    def transform(self, X):
        """
        Project images onto eigenface space
        
        Args:
            X: Images of shape (n_samples, n_features)
            
        Returns:
            Projected coordinates of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_face
        return X_centered @ self.eigenfaces
    
    def inverse_transform(self, proj):
        """
        Reconstruct images from eigenface space
        
        Args:
            proj: Projected coordinates of shape (n_samples, n_components)
            
        Returns:
            Reconstructed images of shape (n_samples, n_features)
        """
        return self.mean_face + proj @ self.eigenfaces.T
    
    def reconstruct(self, X):
        """Full pipeline: project then reconstruct"""
        proj = self.transform(X)
        return self.inverse_transform(proj)


class EigenfaceRecognizer:
    """Face recognizer using eigenfaces and nearest neighbor"""
    
    def __init__(self, pca, threshold=None):
        self.pca = pca
        self.threshold = threshold
        self.train_proj = None
        self.train_labels = None
    
    def fit(self, X_train, y_train):
        """Train the recognizer"""
        self.train_proj = self.pca.transform(X_train)
        self.train_labels = y_train
        
        if self.threshold is None:
            self.threshold = self._auto_compute_threshold()
            print(f"Auto-set confidence threshold: {self.threshold:.4f}")
        
        return self
    
    def _auto_compute_threshold(self):
        """Compute optimal threshold based on training distances"""
        distances = []
        n_samples = self.train_proj.shape[0]
        
        
        sample_size = min(500, n_samples * n_samples)
        
        for i in range(min(50, n_samples)):  # Limit computation
            for j in range(i+1, min(100, n_samples)):
                if self.train_labels[i] == self.train_labels[j]:
                    dist = np.linalg.norm(self.train_proj[i] - self.train_proj[j])
                    distances.append(dist)
        
        if len(distances) > 0:
            return np.percentile(distances, 90)
        else:
            return 1.0  # Default threshold
    
    def predict(self, X_test, return_distances=False):
        """Predict labels for test images"""
        test_proj = self.pca.transform(X_test)
        
        predictions = []
        distances = []
        
        for i in range(test_proj.shape[0]):
            # Compute distances to all training images
            dists = np.linalg.norm(self.train_proj - test_proj[i], axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            
            distances.append(min_dist)
            
            if min_dist < self.threshold:
                predictions.append(self.train_labels[min_idx])
            else:
                predictions.append(-1)  # Unknown face
        
        if return_distances:
            return np.array(predictions), np.array(distances)
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """Compute accuracy (ignoring unknowns)"""
        preds = self.predict(X_test)
        known_mask = preds != -1
        if np.sum(known_mask) > 0:
            return accuracy_score(y_test[known_mask], preds[known_mask])
        return 0.0

def run_eigenface_pipeline(X, y):
    """
    Complete eigenface pipeline
    """
    print("\n" + "="*60)
    print("EIGENFACE RECOGNITION PIPELINE")
    print("="*60)
    
    # Step 1: Split data
    print("\n[Step 1] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Training: {X_train.shape[0]} images")
    print(f"Testing: {X_test.shape[0]} images")
    
    # Step 2: Try different numbers of components
    print("\n[Step 2] Testing different numbers of eigenfaces...")
    n_components_list = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    accuracies = []
    explained_variances = []
    
    for k in n_components_list:
        print(f"\n--- Testing k={k} ---")
        
        # Fit PCA
        pca = EigenfacePCA(n_components=min(k, X_train.shape[0] - 1))
        start_time = time.time()
        pca.fit(X_train)
        fit_time = time.time() - start_time
        
        # Train recognizer
        recognizer = EigenfaceRecognizer(pca)
        recognizer.fit(X_train, y_train)
        
        # Evaluate
        acc = recognizer.score(X_test, y_test)
        accuracies.append(acc)
        explained_variances.append(np.sum(pca.explained_variance_ratio))
        
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Variance explained: {np.sum(pca.explained_variance_ratio)*100:.2f}%")
        print(f"   Fit time: {fit_time:.2f}s")
    
    # Step 3: Find best k
    best_idx = np.argmax(accuracies)
    best_k = n_components_list[best_idx]
    best_acc = accuracies[best_idx]
    
    
    print(f"✅ BEST RESULT: {best_k} eigenfaces → {best_acc*100:.2f}% accuracy")
   
    
    # Step 4: Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs components
    axes[0].plot(n_components_list, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc*100:.1f}%')
    axes[0].set_xlabel('Number of Eigenfaces (k)', fontsize=12)
    axes[0].set_ylabel('Recognition Accuracy', fontsize=12)
    axes[0].set_title('Dimensionality Reduction vs Classification Performance', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Explained variance
    axes[1].plot(n_components_list, explained_variances, 'gs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Eigenfaces (k)', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Variance Captured by Eigenfaces', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_eigenfaces.png', dpi=150)
    plt.show()
    
    return best_k, pca, recognizer


def visualize_eigenfaces(pca, img_shape, n_faces=16):
    """Display top eigenfaces"""
    n_cols = 4
    n_rows = min(n_faces // n_cols + 1, 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.ravel()
    
    for i in range(min(n_faces, pca.eigenfaces.shape[1])):
        eigenface = pca.eigenfaces[:, i].reshape(img_shape)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}')
        axes[i].axis('off')
    
    for i in range(min(n_faces, pca.eigenfaces.shape[1]), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Top Eigenfaces - Principal Facial Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('eigenfaces.png', dpi=150)
    plt.show()


def visualize_mean_face(pca, img_shape):
    """Display the mean face"""
    plt.figure(figsize=(6, 8))
    mean_face = pca.mean_face.reshape(img_shape)
    plt.imshow(mean_face, cmap='gray')
    plt.title('Mean Face (Average of all training faces)', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mean_face.png', dpi=150)
    plt.show()


def visualize_reconstruction(pca, original_images, img_shape, n_samples=3, k_values=[10, 30, 60, 100]):
    """Show original vs reconstructed faces with different k"""
    fig, axes = plt.subplots(len(k_values) + 1, n_samples, figsize=(n_samples*3, (len(k_values)+1)*3))
    
    
    for col in range(n_samples):
        axes[0, col].imshow(original_images[col].reshape(img_shape), cmap='gray')
        axes[0, col].set_title('Original')
        axes[0, col].axis('off')
    
    for row, k in enumerate(k_values):
       
        pca_k = EigenfacePCA(n_components=k)
        pca_k.fit(pca.train_data) 
        
        for col in range(n_samples):
           
            pass
    
    plt.suptitle('Face Reconstruction with Varying Number of Eigenfaces', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
   
    path = r"C:\Users\MURALI KRISHNA\Downloads\archive"
    X, y = load_yale_dataset(path)
    
    print(f"Loaded {X.shape[0]} images, each with {X.shape[1]} pixels")
    print(f"Subjects: 0 to {max(y)}")
    
   
    best_k, pca, recognizer = run_eigenface_pipeline(X, y)
    
    
    img_shape = (243, 320) 
    
    print("\n[Visualization] Mean face...")
    visualize_mean_face(pca, img_shape)
    
    print("[Visualization] Eigenfaces...")
    visualize_eigenfaces(pca, img_shape, n_faces=16)
    
   