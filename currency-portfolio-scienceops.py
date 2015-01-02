from cvxopt import matrix, solvers
from operator import itemgetter
from pandas import DataFrame, concat
from yhat import Yhat, YhatModel, preprocess
import numpy
import csv

csv_rdr = csv.reader(open('2014-12-28-fed-monthly-currency-data.csv'))
raw_data = [row for row in csv_rdr]

# Transpose header metadata and turn it into a data frame
head_data = [[] for _ in range(len(raw_data[0]))]
for row in raw_data[:6]:
    for j, val in enumerate(row):
        head_data[j].append(val)

metadata = DataFrame.from_records(data=head_data[1:], columns=head_data[0])

# Filter out NA currencies
metadata = metadata[metadata['Currency:'] != 'NA']

# This will have the same indices as the rest of the metadata.
countries = []
for c in metadata['Series Description']:
    c = c.upper()
    c = c.split(' -- ')[0]
    c = c.split(' - ')[0]
    countries.append(c)
countries[1] = 'EURO AREA'

# Read in the exchange rates.
exchange_rates = DataFrame.from_records(data=raw_data[6:], columns=raw_data[5])
exchange_rates = exchange_rates[['Time Period'] + list(metadata['Time Period'])]

# Convert the exchange rate data frame into percentage returns.
rows = []
for i in range(len(exchange_rates)-1):
    row = {}
    for tp, cur in zip(metadata['Time Period'], metadata['Currency:']):
        x1 = float(exchange_rates[tp][i])
        x2 = float(exchange_rates[tp][i+1])

        if cur == 'USD':
            x1 = 1.0 / x1
            x2 = 1.0 / x2

        # Returns are in units of %.
        row[tp] = 100 * (x1 - x2) / x2
    rows.append(row)

returns = DataFrame(data=rows, columns=list(metadata['Time Period']))
returns_cov = returns.cov()

# Means are the expected returns for each currency.
exp_returns =  concat({'mean': returns.mean(), 'variance': returns.var()}, axis = 1)

class CurrencyPortfolio(YhatModel):
    @preprocess(in_type=dict, out_type=dict)
    def execute(self, data):
        P = matrix(data['alpha'] * returns_cov.as_matrix())
        q = matrix(-exp_returns['mean'].as_matrix())
        G = matrix(0.0, (len(q),len(q)))
        G[::len(q)+1] = -1.0
        h = matrix(0.0, (len(q),1))
        A = matrix(1.0, (1,len(q)))
        b = matrix(1.0)

        solution = solvers.qp(P, q, G, h, A, b)
        expected_return = exp_returns['mean'].dot(solution['x'])[0]
        variance = sum(solution['x'] * returns_cov.as_matrix().dot(solution['x']))[0]

        investments = {}
        for i, amount in enumerate(solution['x']):
            # Ignore values that appear to have converged to 0.
            if amount > 10e-5:
                investments[countries[i]] = amount*100

        return {
            'alpha': data['alpha'],
            'investments': investments,
            'expected_return': expected_return,
            'variance': variance
        }

yh = Yhat('USERNAME', 'APIKEY', 'http://cloud.yhathq.com/')
yh.deploy('CurrencyPortfolio', CurrencyPortfolio, globals())
