import pandas as pd


def process_bfe_data(data_dir):
    data_file_lines = []

    with open(data_dir+'forecast_error/noaa_best_forecast_error.txt') as f:
        data_file_lines = f.readlines()

    column_labels = []
    cleaned_data_file_lines = []

    #line 7 is where actual data starts, may need to change if txt file changes
    for i, line in enumerate(data_file_lines[7:]):
        line_data = line.rstrip('\n')
        line_data = line_data.split()
        line_data = list(map(str.strip, line_data))

        if i == 0:
            column_labels = line_data
        else:
            cleaned_data_file_lines.append(line_data)

    df = pd.DataFrame(cleaned_data_file_lines, columns=column_labels)

    droplist = [i for i in df.columns if i.endswith('02')]  #gets rid of BCD5 model forecast errors
    df.drop(droplist, axis=1, inplace=True)


    print("#### Saving processed data to file ####")
    df.to_csv(data_dir+'forecast_error/bfe_processed.csv', index = False)
    print("Done")


if __name__ == "__main__":
    DATA_DIR = "./data/"
    process_bfe_data(DATA_DIR)
