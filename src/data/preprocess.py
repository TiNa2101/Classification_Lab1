import pandas as pd
import numpy as np
import src.config as cfg
from src.config import TARGET_COLS


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def fill_smoke_freq(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.SMOKE_FREQ] = df[cfg.SMOKE_FREQ].fillna('0')
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


gender = {
    'Ж': 0, 'М': 1
}
def gender_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.SEX_COL] = df[cfg.SEX_COL].map(gender).astype(np.int8)
    return df


family = {
    'никогда не был(а) в браке': 0, 'в браке в настоящее время': 1, 'гражданский брак / проживание с партнером': 2, 'в разводе': 3, 'вдовец / вдова': 4, 'раздельное проживание (официально не разведены)': 5
}
def family_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Семья'] = df['Семья'].map(family).astype(np.int8)
    return df


ethnos = {
    'другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)': 0, 'европейская': 1, 'прочее (любая иная этно-расовая группа, не представленная выше)': 2
}
def ethnos_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Этнос'] = df['Этнос'].map(ethnos).astype(np.int8)
    return df


nation = {
    'Русские': 0, 'Татары': 1, 'Немцы': 2, 'Азербайджанцы': 3, 'Эстонцы': 4, 'Молдаване': 5, 'Украинцы':6, 'Чуваши': 7, 'Мордва': 8, 'Киргизы': 9, 'Казахи': 10, 'Армяне': 11, 'Белорусы': 12, 'Таджики': 13, 'Башкиры': 14, 'Евреи': 15, 'Буряты': 16, 'Удмурты': 17, 'Лезгины':18, 'Другие национальности': 19
}
def nation_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Национальность'] = df['Национальность'].map(nation).astype(np.int8)
    return df


religion = {
    'Атеист / агностик': 0, 'Христианство': 1, 'Ислам': 2, 'Индуизм': 3, 'Другое': 4, 'Нет': 5
}
def religion_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Религия'] = df['Религия'].map(religion).astype(np.int8)
    return df


def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.EDU_COL] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df


profession = {
    'ведение домашнего хозяйства': 0, 'служащие': 1, 'работники,  занятые в сфере обслуживания, торговые работники магазинов и рынков': 2, 'низкоквалифицированные работники': 3, 'дипломированные специалисты': 4, 'операторы и монтажники установок и машинного оборудования': 5, 'представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры': 6, 'техники и младшие специалисты': 7, 'квалифицированные работники сельского хозяйства и рыболовного': 8, 'ремесленники и представители других отраслей промышленности': 9, 'вооруженные силы': 10
}
def profession_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Профессия'] = df['Профессия'].map(profession).astype(np.int8)
    return df


param_smoke = {
    'Курит': 0, 'Бросил(а)': 1, 'Никогда не курил(а)': 2, 'Никогда не курил': 3
}
def param_smoke_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Статус Курения'] = df['Статус Курения'].map(param_smoke).astype(np.int8)
    return df


passive_smoking_frequency = {
    '0': 0, 'не менее 1 раза в день': 1, '2-3 раза в день': 2, '4 и более раз в день': 3, '1-2 раза в неделю': 4, '3-6 раз в неделю': 5
}
def passive_smoking_frequency_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.SMOKE_FREQ] = df[cfg.SMOKE_FREQ].map(passive_smoking_frequency).astype(np.int8)


alcohol = {
    'никогда не употреблял': 0, 'ранее употреблял': 1, 'употребляю в настоящее время': 2
}
def alcohol_to_num(df: pd.DataFrame) -> pd.DataFrame:
    df['Алкоголь'] = df['Алкоголь'].map(alcohol).astype(np.int8)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)

    df = gender_to_num(df)
    df = family_to_num(df)
    df = ethnos_to_num(df)
    df = nation_to_num(df)
    df = religion_to_num(df)
    df = add_ord_edu(df)
    df = profession_to_num(df)
    df = param_smoke_to_num(df)
    df = passive_smoking_frequency_to_num(df)
    df = alcohol_to_num(df)
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[TARGET_COLS]
    return df, target
