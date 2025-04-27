import os
import pandas as pd

input_csv = "data/data.csv"
pre_path = "../../../../../../"  # "/projects/ceib/data_picai/"

df = pd.read_csv(os.path.join("../../../../../", input_csv))

df_case5 = df[df["case_ISUP"] == 5]
df_case0 = df[df["case_ISUP"] == 0]

if df_case5.empty or df_case0.empty:
    raise ValueError("No se encontraron casos con case_ISUP 5 o 0 en el CSV.")

row5 = df_case5.iloc[0]
row0 = df_case0.iloc[0]

def crear_csv(modality: str, col_image: str, output_filename: str):
    data = {
        "Image": [
            os.path.join(pre_path, row5[col_image]),
            os.path.join(pre_path, row0[col_image])
        ],
        "Mask": [
            os.path.join(pre_path, row5["whole_gland_path"]),
            os.path.join(pre_path, row0["whole_gland_path"])
        ]
    }
    df_mod = pd.DataFrame(data)
    df_mod.to_csv(output_filename, index=False)
    print(f"Archivo '{output_filename}' generado para la modalidad {modality}.")

crear_csv("T2", "t2w_path", "./CSVs/secuencia_T2.csv")
crear_csv("ADC", "adc_path", "./CSVs/secuencia_ADC.csv")
crear_csv("DWI", "hbv_path", "./CSVs/secuencia_DWI.csv")

with open("./CSVs/info_modalidades.txt", "w") as f:
    f.write(f"Fila 1: patient_id: {row5['patient_id']}, study_id: {row5['study_id']}, case_ISUP: {row5['case_ISUP']}\n")
    f.write(f"Fila 2: patient_id: {row0['patient_id']}, study_id: {row0['study_id']}, case_ISUP: {row0['case_ISUP']}\n\n")

print("Archivo 'info_modalidades.txt' generado con la información de patient_id y study_id para cada fila.")