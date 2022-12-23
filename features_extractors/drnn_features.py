import numpy as np
from corenlp.deptree import DependencyTree
from utils.embedding_utils import create_vectorizer, map_to_indices



class DRNNFeatureExtractor:

    def __init__(self, word_to_index, pos_tag_to_index, relation_to_index, lemma_to_index, max_length):
        self.word_vectorizer = create_vectorizer(word_to_index, max_length)
        self.pos_tag_vectorizer = create_vectorizer(pos_tag_to_index, max_length)
        self.relation_vectorizer = create_vectorizer(relation_to_index, max_length)
        self.lemma_vectorizer = create_vectorizer(lemma_to_index, max_length)
        self.max_length = max_length
            

    def compute_features(self, data, labels):
        features = []
        final_labels = []
        for sentence_index, label in zip(data, labels):
            sentence_data = data[sentence_index]
            sample_features = self.__compute_sample_features(sentence_data)
            
            if sample_features is None:
                continue

            features.append(sample_features)
            final_labels.append(label)

        return np.array(features), np.array(final_labels)


    def __compute_sample_features(self, sentence_data):
        dep_tree = DependencyTree(sentence_data['sentence'], sentence_data['entity_1'], sentence_data['entity_2'])
        e1_path, e2_path = dep_tree.get_entities_dependency_paths()
        
        if e1_path is None and e2_path is None:
            return None

        e1_path_features = self.__compute_path_features(e1_path, sentence_data['words'])
        e2_path_features = self.__compute_path_features(e2_path, sentence_data['words'])
        return np.stack([e1_path_features, e2_path_features], axis=0)


    def __compute_path_features(self, path_nodes, tokens_features):
        words = []
        pos_tags = []
        relations = []
        lemmas = []

        for node in path_nodes:
            token_features = tokens_features[node-1]
            words.append(token_features['lc_form'])
            pos_tags.append(token_features['pos'])
            relations.append(token_features['rel'])
            lemmas.append(token_features['lemma'])

        word_indices = map_to_indices(self.word_vectorizer, words)
        pos_tag_indices = map_to_indices(self.pos_tag_vectorizer, pos_tags)
        relation_indices = map_to_indices(self.relation_vectorizer, relations)
        lemma_indices = map_to_indices(self.lemma_vectorizer, lemmas)

        return np.stack([word_indices, pos_tag_indices, relation_indices, lemma_indices], axis=1)