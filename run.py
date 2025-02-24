import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from utils.data_utils import load_data, preprocess_data, create_qa_data
from utils.model_utils import initialize_model, create_vector_store, create_qa_chain


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """
    변경사항 1. 변수명 수정
        - train -> train_df
        - test -> test_df
        - combined_train_data -> train_data
        - combined_test_data -> test_data

    변경사항 2. Inference 기본 코드 수정
        - 일정 idx마다 출력되는 것 대신 tqdm 사용
        - test_data를 iterrows()로 순회하는 대신 itertuples()로 순회
        - for문을 사용하여 test_data를 순회하며 결과를 저장하는 대신 list comprehension 사용
    """
    # load config
    cfg = load_config(arg_parser().cfg_path)
    train_data_path = cfg["paths"]["train_data"]
    test_data_path = cfg["paths"]["test_data"]
    submission_path = cfg["paths"]["submission"]
    output_path = cfg["paths"]["output"]
    model_name = cfg["model"]["model_name"]
    model_path = cfg["model"]["model_path"]
    embedding_model_name = cfg["model"]["embedding_model"]
    prompt_template = cfg["prompt_template"]
    batch_size = cfg["settings"]["batch_size"]
    max_new_tokens = cfg["settings"]["max_new_tokens"]
    search_model = cfg["settings"]["search_model"]
    search_k_num = cfg["settings"]["search_k_num"]

    # load data
    train_df, test_df = load_data(train_data_path, test_data_path)
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    train_data = create_qa_data(train_df, is_train=True)
    test_data = create_qa_data(test_df, is_train=False)

    # Import model
    tokenizer, model = initialize_model(model_name, model_path)

    # Create vector store
    vector_store = create_vector_store(train_data, embedding_model_name)

    # Generate RAG chain
    qa_chain = create_qa_chain(
        vector_store,
        model,
        tokenizer,
        prompt_template,
        search_model,
        search_k_num,
        max_new_tokens,
    )

    # Batch processing
    test_dataset = Dataset.from_pandas(test_data)

    # Inference
    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test_data))
    test_results = [
        qa_chain.invoke(row.question)["result"]
        for row in tqdm(
            test_data.itertuples(index=False), total=len(test_data), desc="Processing"
        )
    ]

    # Submission
    embedding = SentenceTransformer(embedding_model_name)
    pred_embeddings = embedding.encode(test_results)

    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    submission.iloc[:, 1] = test_results
    submission.iloc[:, 2:] = pred_embeddings
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
