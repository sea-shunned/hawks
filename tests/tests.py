import unittest
import sys
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
sys.path.append('..')
import hawks
from hawks.cluster import Cluster
from hawks.dataset import Dataset
from hawks.genotype import Genotype

class OperatorTests(unittest.TestCase):
    # Test mutation and crossover operators
    def setUp(self):
        rng = np.random.RandomState(42)
        Cluster.global_rng = rng
        Genotype.global_rng = rng

        setattr(Cluster, "num_dims", 2)
        setattr(Cluster, "initial_mean_upper", 1.0)
        setattr(Cluster, "initial_cov_upper", 0.5)

        clust1 = Cluster(70)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[1, 0], [0, 1]])
        clust2 = Cluster(90)
        clust2.mean = np.array([5, 5])
        clust2.cov = np.array([[5, 0], [0, 10]])
        self.indiv1 = Genotype([clust1, clust2])
        self.indiv1.create_views()
        self.indiv1.resample_values()

        clust3 = Cluster(70)
        clust3.mean = np.array([2, 2])
        clust3.cov = np.array([[2, 0], [0, 2]])
        clust4 = Cluster(90)
        clust4.mean = np.array([10, 10])
        clust4.cov = np.array([[4, 0], [0, 2]])
        self.indiv2 = Genotype([clust3, clust4])
        self.indiv2.create_views()
        self.indiv2.resample_values()

    def tearDown(self):
        Cluster.global_rng = None
        Genotype.global_rng = None

        delattr(Cluster, "num_dims")

    def test_uniform_crossover_genes(self):
        # Get the sequence of the numbers to be generated
        rng = np.random.RandomState(42)
        swaps = [True if rng.rand() < 0.5 else False for _ in range(len(self.indiv1)*2)]

        indiv1_means = [i.mean for i in self.indiv1]
        indiv2_means = [i.mean for i in self.indiv2]

        indiv1_covs = [i.cov for i in self.indiv1]
        indiv2_covs = [i.cov for i in self.indiv2]

        Genotype.global_rng = np.random.RandomState(42)
        self.indiv1, self.indiv2 = Genotype.xover_genes(
            self.indiv1,
            self.indiv2,
            mixing_ratio=0.5
        )

        # Test means
        for i, (clust1, clust2, swap) in enumerate(zip(self.indiv1, self.indiv2, swaps[0::2])):
            with self.subTest(i=i):
                if swap:
                    self.assertTrue(np.array_equal(clust1.mean, indiv2_means[i]))
                    self.assertTrue(np.array_equal(clust2.mean, indiv1_means[i]))
                else:
                    self.assertTrue(np.array_equal(clust1.mean, indiv1_means[i]))
                    self.assertTrue(np.array_equal(clust2.mean, indiv2_means[i]))
        # Test covs
        for i, (clust1, clust2, swap) in enumerate(zip(self.indiv1, self.indiv2, swaps[1::3])):
            with self.subTest(i=i):
                if swap:
                    self.assertTrue(np.array_equal(clust1.cov, indiv2_covs[i]))
                    self.assertTrue(np.array_equal(clust2.cov, indiv1_covs[i]))
                else:
                    self.assertTrue(np.array_equal(clust1.cov, indiv1_covs[i]))
                    self.assertTrue(np.array_equal(clust2.cov, indiv2_covs[i]))

    def test_uniform_crossover_clusters(self):
        rng = np.random.RandomState(42)
        swaps = [True if rng.rand() < 0.5 else False for _ in self.indiv1]

        indiv1_ids = [id(i) for i in self.indiv1]
        indiv2_ids = [id(i) for i in self.indiv2]

        Genotype.global_rng = np.random.RandomState(42)
        self.indiv1, self.indiv2 = Genotype.xover_cluster(
            self.indiv1,
            self.indiv2
        )
        # Test whether cluster objects have been swapped or not
        for i, (clust1, clust2, swap) in enumerate(zip(self.indiv1, self.indiv2, swaps)):
            with self.subTest(i=i):
                if swap:
                    self.assertEqual(indiv1_ids[i], id(clust2))
                    self.assertEqual(indiv2_ids[i], id(clust1))
                else:
                    self.assertEqual(indiv1_ids[i], id(clust1))
                    self.assertEqual(indiv2_ids[i], id(clust2))

    def test_uniform_crossover_none(self):
        self.indiv1, self.indiv2 = Genotype.xover_genes(
            self.indiv1,
            self.indiv2,
            mixing_ratio=0.0
        )

        indiv1_unchanged = all([
            np.allclose(
                self.indiv1[0].mean,
                np.array([0, 0])
            ),
            np.allclose(
                self.indiv1[0].cov,
                np.array([[1, 0], [0, 1]])
            ),
            np.allclose(
                self.indiv1[1].mean,
                np.array([5, 5])
            ),
            np.allclose(
                self.indiv1[1].cov,
                np.array([[5, 0], [0, 10]])
            )
        ])

        self.assertTrue(indiv1_unchanged)

    def test_uniform_crossover_all(self):
        self.indiv1, self.indiv2 = Genotype.xover_genes(
            self.indiv1,
            self.indiv2,
            mixing_ratio=1.0
        )

        indiv1_allchanged = not any([
            np.allclose(
                self.indiv1[0].mean,
                np.array([0, 0])
            ),
            np.allclose(
                self.indiv1[0].cov,
                np.array([[1, 0], [0, 1]])
            ),
            np.allclose(
                self.indiv1[1].mean,
                np.array([5, 5])
            ),
            np.allclose(
                self.indiv1[1].cov,
                np.array([[5, 0], [0, 10]])
            )
        ])

        self.assertTrue(indiv1_allchanged)

class GenotypeTests(unittest.TestCase):
    # Test some methods of the Genotype - mainly the array view manipulation
    def setUp(self):
        rng = np.random.RandomState(42)
        Cluster.global_rng = rng
        Genotype.global_rng = rng

        setattr(Cluster, "num_dims", 2)
        setattr(Cluster, "initial_mean_upper", 1.0)
        setattr(Cluster, "initial_cov_upper", 0.5)

        clust1 = Cluster(50)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[1, 0], [0, 1]])
        clust2 = Cluster(30)
        clust2.mean = np.array([5, 5])
        clust2.cov = np.array([[5, 0], [0, 10]])
        self.indiv1 = Genotype([clust1, clust2])
        self.indiv1.create_views()
        self.indiv1.resample_values()

        clust3 = Cluster(50)
        clust3.mean = np.array([2, 2])
        clust3.cov = np.array([[2, 0], [0, 2]])
        clust4 = Cluster(30)
        clust4.mean = np.array([10, 10])
        clust4.cov = np.array([[4, 0], [0, 2]])
        self.indiv2 = Genotype([clust3, clust4])
        self.indiv2.create_views()
        self.indiv2.resample_values()

    def tearDown(self):
        Cluster.global_rng = None
        Genotype.global_rng = None

        delattr(Cluster, "num_dims")

    def test_views_swap_all_genes(self):
        orig_all_vals = np.copy(self.indiv2.all_values)

        self.indiv1, self.indiv2 = Genotype.xover_genes(
            self.indiv1,
            self.indiv2,
            mixing_ratio=1.0
        )

        self.indiv1.recreate_views()
        self.indiv1.resample_values()

        close = np.allclose(self.indiv1.all_values, orig_all_vals)

        self.assertTrue(close)

    def test_views_swap_all_clusters(self):
        orig_all_vals = np.copy(self.indiv2.all_values)

        self.indiv1, self.indiv2 = Genotype.xover_cluster(
            self.indiv1,
            self.indiv2,
            mixing_ratio=1.0
        )

        self.indiv1.recreate_views()
        self.indiv1.resample_values()

        close = np.allclose(self.indiv1.all_values, orig_all_vals)

        self.assertTrue(close)

class ConstraintTests(unittest.TestCase):
    # Test constraints
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        Cluster.global_rng = rng
        Genotype.global_rng = rng

        setattr(Cluster, "num_dims", 2)
        setattr(Cluster, "initial_mean_upper", 1.0)
        setattr(Cluster, "initial_cov_upper", 0.5)

    @classmethod
    def tearDownClass(cls):
        Cluster.global_rng = None
        Genotype.global_rng = None

        delattr(Cluster, "num_dims")

    def test_overlap_separated(self):
        clust1 = Cluster(50)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[1, 0], [0, 1]])

        clust2 = Cluster(80)
        clust2.mean = np.array([10, 10])
        clust2.cov = np.array([[1, 0], [0, 1]])

        indiv = Genotype([clust1, clust2])
        indiv.create_views()
        indiv.resample_values()
        
        overlap = hawks.constraints.overlap(indiv)

        self.assertEqual(overlap, 0)

    def test_overlap_same(self):
        clust1 = Cluster(4000)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[1, 0], [0, 1]])

        clust2 = Cluster(4000)
        clust2.mean = np.array([0, 0])
        clust2.cov = np.array([[1, 0], [0, 1]])

        indiv = Genotype([clust1, clust2])
        indiv.create_views()
        indiv.resample_values()

        overlap = hawks.constraints.overlap(indiv)

        self.assertAlmostEqual(overlap, 0.5, places=1)

    def test_eigenratio_spherical(self):
        clust1 = Cluster(50)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[1, 0], [0, 1]])

        clust2 = Cluster(80)
        clust2.mean = np.array([0, 0])
        clust2.cov = np.array([[1, 0], [0, 1]])

        indiv = Genotype([clust1, clust2])
        eigen_ratio = hawks.constraints.eigenval_ratio(indiv)

        self.assertEqual(eigen_ratio, 1)

    def test_eigenratio_eccentric(self):
        clust1 = Cluster(50)
        clust1.mean = np.array([0, 0])
        clust1.cov = np.array([[10, 0], [0, 1]])

        clust2 = Cluster(80)
        clust2.mean = np.array([0, 0])
        clust2.cov = np.array([[9.9, 0], [0, 1]])

        indiv = Genotype([clust1, clust2])
        eigen_ratio = hawks.constraints.eigenval_ratio(indiv)        

        self.assertEqual(eigen_ratio, 10)

class EvolutionaryTests(unittest.TestCase):
    # **TODO** Some tests for different selection methods

    def setUp(self):
        self.gen = hawks.create_generator("validation.json")
        self.init_pop = []
        for indiv in self.gen.create_individual():
            self.init_pop.append(indiv)

    def tearDown(self):
        del self.gen

    def test_overlap_consistent(self):
        pop = hawks.ga.generation(self.init_pop, self.gen.deap_toolbox, self.gen.full_config["constraints"], cxpb=0.7)

        overlaps_before = np.sum([indiv.constraints["overlap"] for indiv in pop])
        for indiv in pop:
            indiv.calc_constraints(self.gen.full_config["constraints"])
        overlaps_after = np.sum([indiv.constraints["overlap"] for indiv in pop])
        self.assertAlmostEqual(overlaps_before, overlaps_after)

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some baseline args so we can just modify the relevant onesdef tearDown(self):self):
        cls.args = {
            "num_examples": 1000,
            "num_clusters": 10,
            "num_dims": 2,
            "equal_clusters": False,
            "min_clust_size": 5
        }

        Dataset.global_rng = np.random.RandomState(42)

    @classmethod
    def tearDownClass(cls):
        Dataset.global_rng = None

    def test_equal_clusters_are_equal(self):
        # test that the cluster sizes are equal and expected
        kwargs = self.args.copy()
        # Pick tricky numbers
        kwargs["num_examples"] = 7883
        kwargs["num_clusters"] = 17
        kwargs["equal_clusters"] = True
        obj = Dataset(**kwargs)

        for size in obj.cluster_sizes:
            with self.subTest(size=size):
                self.assertEqual(size, 464)
    
    def test_random_cluster_sizes_sum(self):
        # Have a loop that tests different sizes and that we generate something close to that)
        kwargs = self.args.copy()
        kwargs["num_examples"] = 10000
        kwargs["num_clusters"] = 100
        obj = Dataset(**kwargs)

        sizes_sum = np.sum(obj.cluster_sizes)

        close_to_sum = np.around(sizes_sum, decimals=-1) == kwargs["num_examples"]

        self.assertTrue(close_to_sum)

    def test_random_cluster_sizes_different(self):
        kwargs = self.args.copy()
        kwargs["num_examples"] = 10000
        kwargs["num_clusters"] = 100
        obj = Dataset(**kwargs)

        unique_sizes = set()
        for size in obj.cluster_sizes:        
            unique_sizes.add(size)
        # There is a chance of a clash, but ensure not all are the same
        self.assertTrue(len(unique_sizes) > 1)

    def test_incorrect_min_clust_size(self):
        kwargs = self.args.copy()
        kwargs["num_examples"] = 100
        kwargs["num_clusters"] = 10
        kwargs["min_clust_size"] = 20 # Not possible, will be modified
        obj = Dataset(**kwargs)

        self.assertLessEqual(obj.min_clust_size, obj.num_examples/obj.num_clusters)
    
    def test_exact_min_clust_size(self):
        kwargs = self.args.copy()
        kwargs["num_examples"] = 200
        kwargs["num_clusters"] = 5
        kwargs["min_clust_size"] = 40 # Not possible, will be modified
        obj = Dataset(**kwargs)

        self.assertEqual(obj.cluster_sizes, [40]*5)
    # **TODO** add a test for size tuple method if we add it

class ObjectiveTests(unittest.TestCase):
    # Test that the objectives return as they should
    def setUp(self):
        # Whatever setup is needed
        rng = np.random.RandomState(42)
        Genotype.global_rng = rng
        Cluster.global_rng = rng
        sizes = [190, 20, 30, 110]
        self.indiv = Genotype([Cluster(size) for size in sizes])
        self.indiv.create_views()
        self.indiv.resample_values()
        hawks.objectives.Silhouette.setup_indiv(self.indiv)

    def tearDown(self):
        self.indiv = None
        rng = None
        Genotype.global_rng = None
        Cluster.global_rng = None

    def test_silhouette_complete_indiv(self):
        hawks.objectives.Silhouette.eval_objective(self.indiv)

        close_to_sk = np.isclose(silhouette_score(
            self.indiv.all_values,
            self.indiv.labels,
            metric="sqeuclidean"), self.indiv.silhouette)
        
        self.assertTrue(close_to_sk)

    def test_silhouette_changed_indiv(self):
        hawks.objectives.Silhouette.eval_objective(self.indiv)
        for cluster in self.indiv:
            cluster.changed = False
        
        self.indiv[0].gen_initial_mean()
        self.indiv[0].changed = True
        aux = self.indiv.silhouette
        
        self.indiv.resample_values()
        hawks.objectives.Silhouette.eval_objective(self.indiv)
        
        self.assertNotAlmostEqual(aux, self.indiv.silhouette)

        close_to_sk = np.isclose(silhouette_score(
            self.indiv.all_values,
            self.indiv.labels,
            metric="sqeuclidean"), self.indiv.silhouette)
        
        self.assertTrue(close_to_sk)
    
    def test_silhouette_singleton_cluster(self):
        rng = np.random.RandomState(42)
        Genotype.global_rng = rng
        Cluster.global_rng = rng
        sizes = [1, 20, 30, 110]
        self.indiv = Genotype([Cluster(size) for size in sizes])
        self.indiv.create_views()
        self.indiv.resample_values()
        hawks.objectives.Silhouette.setup_indiv(self.indiv)

        hawks.objectives.Silhouette.eval_objective(self.indiv)

        close_to_sk = np.isclose(silhouette_score(
            self.indiv.all_values,
            self.indiv.labels,
            metric="sqeuclidean"), self.indiv.silhouette)

        self.assertTrue(close_to_sk)

class HawksTests(unittest.TestCase):
    def test_multiconfig_deep(self):
        config = {
            "dataset": {
                "num_examples": [10, 100, 1000]
            },
            "constraints": {
                "overlap": {
                    "limit": ["upper", "lower"]
                }
            },
            "ga": {
                "num_gens": [50, 100, 10, 200],
                "mut_args_mean": {
                    "dims": ["each", "all"]
                }
            }
        }
        obj = hawks.create_generator(config)
        total_configs, _, _ = obj._count_multiconfigs()
        self.assertEqual(total_configs, 48)

    def test_full_hawks_run(self):
        gen = hawks.create_generator("validation.json")
        gen.run()

        res = gen.get_stats()

        known_result = pd.read_csv(
            "validation.csv",
            index_col=False
        )
        print("Result:")
        print(res)
        print("Known result:")
        print(known_result)
        print("---")
        # Pandas can be iffy with data types
        equals = np.allclose(res.values, known_result.values)
        self.assertTrue(equals)

    def test_full_hawks_run_multiple(self):
        gen = hawks.create_generator("validation.json")
        gen.run()
        # Run a second time to ensure there's no carryover
        gen = hawks.create_generator("validation.json")
        gen.run()

        res = gen.get_stats()

        known_result = pd.read_csv(
            "validation.csv",
            index_col=False
        )
        # Pandas can be iffy with data types
        equals = np.allclose(res.values, known_result.values)
        self.assertTrue(equals)

    def test_incorrect_config_arg(self):
        with self.assertRaises(ValueError):
            gen = hawks.create_generator(
                {
                    "hawks": {
                        "seed_num": 4,
                        "num_runs": 1
                    },
                    "objectives": {
                        "silhouette": {
                            "target": 0.9
                        }
                    },
                    "constraints": {
                        "eigenval_ratio": {
                            "lim": "upper" # <--- error
                        }
                    }
                }
            )

    def test_nested_config_arg(self):
        gen = hawks.create_generator(
            {
                "constraints": {
                    "overlap": {
                        "limit": "TEST"
                    }
                }
            }
        )

        self.assertEqual(
            gen.full_config["constraints"]["overlap"]["limit"],
            "TEST"
        )

if __name__ == '__main__':
    unittest.main(buffer=True)
