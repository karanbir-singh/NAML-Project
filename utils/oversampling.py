import jax
import jax.numpy as jnp
from jax import random
from sklearn import neighbors 
from functools import partial

@partial(jax.jit, static_argnums = (1,))
def generate_synthetic_samples(X_minority, n_samples, indices):
    """
        This is an helper function that is used to generate synthetic samples using the SMOTE algorithm.
        It's JIT-compiled, and the `indices` argument is marked as a static argument for performance.
        
        Args:
            X_minority: array of minority class samples (input feature vectors).
            n_samples: number of synthetic samples to generate.
            indices: array of indices for the nearest neighbors of each minority sample.
            
        Returns:
            A JAX array of synthetic samples.
    """

    key = random.PRNGKey(0)

    n_minority_samples = X_minority.shape[0]

    key, subkey1, subkey2 = random.split(key, 3)

    #Randomly select indices of minority samples to base synthetic samples on
    sample_indices = random.randint(subkey1, (n_samples,), 0, n_minority_samples)

    #Randomly select one of the nearest neighbors for each chosen sample
    neighbor_choices = random.randint(subkey2, (n_samples,), 1, indices.shape[1])

    #"selected samples" are the sample chosen to interpolate from
    selected_samples = X_minority[sample_indices] 

    #Neighbors can be found by indexing the nearest neighbors array with the chosen indices
    selected_neighbors = X_minority[indices[sample_indices, neighbor_choices]]  

    #Generate random interpolation factors (gaps) between 0 and 1
    key, subkey3 = random.split(key)
    gaps = random.uniform(subkey3, (n_samples, 1))

    #Place the syntetic point by interpolating between the selected sample and its neighbor
    diffs = selected_neighbors - selected_samples
    synthetic = selected_samples + gaps * diffs

    return synthetic

def fit_resample(X_minority, n_samples, k = 5):
    """
        Main function to fit the data and generate synthetic samples, using JAX-accelerated SMOTE.
        
        Args:
            X_minority: array of minority class samples (input feature vectors).
            n_samples: number of synthetic samples to generate.
            k: number of nearest neighbors to consider for SMOTE.
        
        Returns:
            A JAX array of synthetic samples: rember that the output is (n_samples, n_features) not (n_smaple + n_minority, n_features),
            so you will need to add later if you nned to.
    """

    X_minority = jnp.array(X_minority)

    #Initialize the NearestNeighbors model from scikit-learn
    #We use "k + 1" because the nearest neighbor should include the sample itself
    #Then, fit the nearest neighbors model on the minority samples
    neigh = neighbors.NearestNeighbors(n_neighbors = k + 1)
    neigh.fit(X_minority) 

    #Find the "k + 1" nearest neighbors for each minority sample
    #"indices" is a 2D array with the indices of the nearest neighbors for each sample
    _, indices = neigh.kneighbors(X_minority)
    indices = jnp.array(indices)

    #Call the synthetic sample generator
    return generate_synthetic_samples(X_minority, n_samples, indices)