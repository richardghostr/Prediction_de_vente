import argparse # sert à lire les arguments depuis le terminal
from pathlib import Path # sert à manipuler les chemins des fichiers
import pandas as pd
required_columns ={"id","date","value"}
def validate_schema(df:pd.DataFrame)-> None:
    missing= required_columns- set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")
def ingest_csv(input_path:Path, output_dest:Path)-> None:
    print(f"Lecture du fichier:{input_path}")
    df=pd.read_csv(input_path)
    validate_schema(df)
    output_dest.mkdir(parents=True,exist_ok=True)
    output_path= output_dest/input_path.name
    df.to_csv(output_path,index=False)
    print(f"fichier validé et ecrit dans: {output_path}")
def main():
    parser=argparse.ArgumentParser(description="ingestion des Csv")
    parser.add_argument(
        "input",
        help="Chemin vers fichier CSV ")
    parser.add_argument(
        "output",
        help="Dossier de sortie"
    )
    args=parser.parse_args()
    input_path=Path(args.input)
    output_dest=Path(args.output)
    if input_path.is_file():
        ingest_csv(input_path,output_dest)
    elif input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files :
            raise FileNotFoundError("Aucun fichier CSV trouvé")
        for csv_file in csv_files:
            ingest_csv(csv_file,output_dest)
    else:
        raise FileNotFoundError(f"Chemin Invalid :{input_path}")

if __name__ == "__main__":
    main()
