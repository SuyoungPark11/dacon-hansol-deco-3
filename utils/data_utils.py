import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    return train, test


def preprocess_data(df):
    df = df.dropna(subset=["인적사고", "사고원인"]).copy()
    df["인적사고"] = df["인적사고"].replace(
        to_replace={r"넘어짐.*": "넘어짐", r"떨어짐.*": "떨어짐"}, regex=True
    )

    for col, delim in [("공사종류", " / "), ("공종", " > "), ("사고객체", " > ")]:
        df[f"{col}(대분류)"] = df[col].str.split(delim).str[0]
        df[f"{col}(중분류)"] = df[col].str.split(delim).str[1]
    return df


def create_qa_data(df, is_train=True):
    qa_data = df.apply(
        lambda row: {
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
            "answer": row["재발방지대책 및 향후조치계획"] if is_train else None,
        },
        axis=1,
    )
    return pd.DataFrame(list(qa_data))
