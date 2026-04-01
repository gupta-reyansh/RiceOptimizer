import os
import pickle
import tempfile
import unittest
import gzip

from CodonTransformer.CodonUtils import (
    IterableJSONData,
    ProteinConfig,
    find_pattern_in_fasta,
    get_organism2id_dict,
    get_taxonomy_id,
    load_pkl_from_url,
    load_python_object_from_disk,
    save_python_object_to_disk,
    sort_amino2codon_skeleton,
)


class TestCodonUtils(unittest.TestCase):
    def test_config_manager(self):
        with ProteinConfig() as config:
            config.set("ambiguous_aminoacid_behavior", "standardize_deterministic")
            self.assertEqual(
                config.get("ambiguous_aminoacid_behavior"), "standardize_deterministic"
            )
            config.set("ambiguous_aminoacid_map_override", {"X": ["A", "G"]})
            self.assertEqual(
                config.get("ambiguous_aminoacid_map_override"), {"X": ["A", "G"]}
            )
            config.update(
                {
                    "ambiguous_aminoacid_behavior": "raise_error",
                    "ambiguous_aminoacid_map_override": {"X": ["A", "G"]},
                }
            )
            self.assertEqual(config.get("ambiguous_aminoacid_behavior"), "raise_error")
            self.assertEqual(
                config.get("ambiguous_aminoacid_map_override"), {"X": ["A", "G"]}
            )
            try:
                config.set("invalid_key", "invalid_value")
                self.fail("Expected ValueError")
            except ValueError:
                pass
        with ProteinConfig() as config:
            self.assertEqual(
                config.get("ambiguous_aminoacid_behavior"), "standardize_random"
            )
            self.assertEqual(config.get("ambiguous_aminoacid_map_override"), {})

    def test_load_python_object_from_disk(self):
        test_obj = {"key1": "value1", "key2": 2}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(test_obj, temp_file_name)
        loaded_obj = load_python_object_from_disk(temp_file_name)
        self.assertEqual(test_obj, loaded_obj)
        os.remove(temp_file_name)

    def test_save_python_object_to_disk(self):
        test_obj = [1, 2, 3, 4, 5]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(test_obj, temp_file_name)
            self.assertTrue(os.path.exists(temp_file_name))
        os.remove(temp_file_name)

    def test_find_pattern_in_fasta(self):
        text = (
            ">seq1 [keyword=value1]\nATGCGTACGTAGCTAG\n"
            ">seq2 [keyword=value2]\nGGTACGATCGATCGAT"
        )
        self.assertEqual(find_pattern_in_fasta("keyword", text), "value1")
        self.assertEqual(find_pattern_in_fasta("nonexistent", text), "")

    def test_get_organism2id_dict(self):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=True, suffix=".csv"
        ) as temp_file:
            temp_file.write("0,Escherichia coli\n1,Homo sapiens\n2,Mus musculus")
            temp_file.flush()
            organism2id = get_organism2id_dict(temp_file.name)
            self.assertEqual(
                organism2id,
                {"Escherichia coli": 0, "Homo sapiens": 1, "Mus musculus": 2},
            )

    def test_get_taxonomy_id(self):
        taxonomy_dict = {
            "Escherichia coli": 562,
            "Homo sapiens": 9606,
            "Mus musculus": 10090,
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(taxonomy_dict, temp_file_name)
            self.assertEqual(get_taxonomy_id(temp_file_name, "Escherichia coli"), 562)
            self.assertEqual(
                get_taxonomy_id(temp_file_name, return_dict=True), taxonomy_dict
            )

    def test_sort_amino2codon_skeleton(self):
        amino2codon = {
            "A": (["GCT", "GCC", "GCA", "GCG"], [0.0, 0.0, 0.0, 0.0]),
            "C": (["TGT", "TGC"], [0.0, 0.0]),
        }
        sorted_amino2codon = sort_amino2codon_skeleton(amino2codon)
        self.assertEqual(
            sorted_amino2codon,
            {
                "A": (["GCA", "GCC", "GCG", "GCT"], [0.0, 0.0, 0.0, 0.0]),
                "C": (["TGC", "TGT"], [0.0, 0.0]),
            },
        )

    def test_load_pkl_from_url(self):
        url = "https://example.com/test.pkl"
        expected_obj = {"key": "value"}
        with unittest.mock.patch("requests.get") as mock_get:
            mock_get.return_value.content = pickle.dumps(expected_obj)
            loaded_obj = load_pkl_from_url(url)
        self.assertEqual(loaded_obj, expected_obj)

    def test_iterable_json_data_reads_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            temp.write('{"idx": 0, "codons": "M_ATG __TAA", "organism": 0}\n')
            temp.write("\n")
            temp.write('{"idx": 1, "codons": "M_ATG K_AAA __TAA", "organism": 0}\n')
            temp_path = temp.name

        try:
            rows = list(IterableJSONData(temp_path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["idx"], 0)
            self.assertEqual(rows[1]["organism"], 0)
        finally:
            os.remove(temp_path)

    def test_iterable_json_data_reads_gzipped_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as temp:
            temp_path = temp.name

        try:
            with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
                handle.write('{"idx": 0, "codons": "M_ATG __TAA", "organism": 0}\n')

            rows = list(IterableJSONData(temp_path))
            self.assertEqual(rows[0]["codons"], "M_ATG __TAA")
        finally:
            os.remove(temp_path)

    def test_iterable_data_defaults_without_slurm_env(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            temp.write('{"idx": 0, "codons": "M_ATG __TAA", "organism": 0}\n')
            temp_path = temp.name

        dataset = IterableJSONData(temp_path, dist_env="slurm")
        worker_info = unittest.mock.Mock(id=1, num_workers=3)

        try:
            with unittest.mock.patch(
                "torch.utils.data.get_worker_info", return_value=worker_info
            ), unittest.mock.patch.dict(os.environ, {}, clear=True):
                rows = list(dataset)
            self.assertEqual(rows, [])
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
