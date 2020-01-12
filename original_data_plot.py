"""
Auxiliary module to generate plots for the memory
"""
import pandas as pd
import matplotlib.pyplot as plt

IMG_FOLDER = 'DATASET_IMAGES/'

if __name__ == "__main__":
    ## Original data plot
    df = pd.read_csv('sp5001962.csv', sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df = df.drop('Open', axis=1)
    df = df.drop('High', axis=1)
    df = df.drop('Low', axis=1)
    df = df.drop('Adj Close', axis=1)
    df = df.drop('Volume', axis=1)

    print(df.loc[df['Close'].idxmax()])
    print(df.loc[df['Close'].idxmin()])
    plt.hist(df['Close'], bins = 20)
    plt.ylabel('Frequency')
    plt.xlabel('Close prices')
    plt.savefig(IMG_FOLDER+'histogram.png')
    plt.clf()

    times = pd.date_range('1962-01-02', '2019-12-05', periods=10)

    plt.figure(num=None, figsize=(16, 9))
    df.set_index(['Date'], inplace=True)
    ax = df.plot()
    ax.get_legend().remove()
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.title('Closing prices of the original dataset')
    plt.xticks(times.to_pydatetime())
    plt.savefig(IMG_FOLDER+'original_data.png')
    plt.clf()

    ## Train-test split plot
    df1 = df.iloc[:11666, :]
    df2 = df.iloc[11666:, :]
    df1.columns = ['Train dataset']
    df2.columns = ['Test dataset']
    ax = plt.gca()
    df1.plot(kind='line', ax=ax, color='blue', label='Train dataset', figsize=(16, 9))
    df2.plot(kind='line', ax=ax, color='red', label='Test dataset')
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.title('Closing prices of the original dataset')
    plt.xticks(times.to_pydatetime())
    plt.legend(loc=2, prop={'size': 14})
    plt.savefig(IMG_FOLDER+'train_test_split.png')
    plt.clf()
