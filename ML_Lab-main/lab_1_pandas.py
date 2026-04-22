import pandas as pd
import numpy as np

def process_storm_data():
    try:
        df = pd.read_csv('cyclone_mocha_data.csv', na_values=[' ', '-', 'N/A'])
        print("--- Data Successfully Imported ---")
        print(df.head())
    except FileNotFoundError:
        data = {
            'Date': ['12.05.23', '12.05.23', '13.05.23'],
            'Time_UTC': ['1800', '2100', '0000'],
            'Lat': [15.1, 15.2, 15.4],
            'Long': [88.8, 88.9, 89.1],
            'Central_Pressure_hPa': [954, 954, 954],
            'Wind_Speed_kt': [100, 100, 100],
            'Grade': ['ESCS', 'ESCS', 'ESCS']
        }
        df = pd.DataFrame(data)
        print("--- Created Sample DataFrame from Scratch ---")

    df['Wind_Speed_kmh'] = df['Wind_Speed_kt'] * 1.852
    df.to_csv('processed_storm_data.csv', index=False)
    df.to_json('storm_api_output.json', orient='records', indent=4)
    df.to_excel('storm_summary_report.xlsx', index=False, sheet_name='Summary')
    print("\n--- Data Successfully Exported to CSV, JSON, and Excel ---")

if __name__ == "__main__":
    process_storm_data()
