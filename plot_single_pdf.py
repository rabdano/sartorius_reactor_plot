import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os
import matplotlib as mpl


mpl.use('Agg')

path_to_file = 'data\Серия 16-21-1\B1.20210520.csv'


print(path_to_file)
filename = path_to_file.split('\\')[-2] + '_' + path_to_file.split('\\')[-1][:-4]
print(filename)
if not os.path.exists('./figures/' + filename):
    os.mkdir('./figures/' + filename)
plot_out_name = './figures/' + filename + '/' + filename + '.pdf'
pp = PdfPages(plot_out_name)
plt.figure(figsize=(16./2.54, 12./2.54))

cols_to_use = [2] + [x for x in range(5, 13, 1)] + [x for x in range(27, 51, 1)] + [60, 61, 62, 63, 64]

data = pd.read_csv(path_to_file,
                   sep=';',
                   header=[0, 1, 2],
                   low_memory=False,
                   parse_dates=[0],
                   dayfirst=True,
                   decimal=',',
                   verbose=True,
                   na_values='BAD',
                   dtype='object')
cols = []
for c in data.columns:
    cols.append(c[0] + ' ' + c[1] + ', ' + c[2])
data.columns = cols
data = data.iloc[:, cols_to_use]

for c in data.columns:
    data[c] = [x.replace(',', '.') if type(x) == str else x for x in data[c]]
    data[c] = pd.to_numeric(data[c], errors='coerce')

time_h = data['ProcessTime Value, hours'].to_numpy(dtype=np.float64)

legends = data.columns
ylims = {'STIRR': [0, 1550],
         'pO2': [0, 100],
         'pH': [6, 8],
         'TURB': [0, 3],
         'TEMP': [20, 45],
         'AIRSP': [0, 11],
         'O2SP': [0, 11],
         'N2SP': [0, 11]}

y_captions = {'STIRR': 'Мешалка, об/мин',
              'pO2': 'Раств. кислород, %',
              'pH': 'pH',
              'TURB': 'Мутность, AU',
              'TEMP': r'Температура, $\degree$С',
              'AIRSP': 'Воздух, л/мин',
              'O2SP': 'Кислород, л/мин',
              'N2SP': 'Азот, л/мин'}

for l in legends:
    # print(l)
    t, y = time_h, data[l].to_numpy()

    plot_df = pd.DataFrame({'time': t, 'y': y}).dropna()

    plt.clf()
    plt.xlim([0, 10])
    for yl in ylims:
        if yl in l:
            plt.ylim(ylims[yl])
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=13)
    plt.plot(plot_df['time'], plot_df['y'], 'k-', lw=0.75)
    plt.xlabel('Время, ч')
    for yc in y_captions:
        if yc in l:
            plt.ylabel(y_captions[yc])
    plt.title(l)
    plt.grid(True, lw=0.3)
    pp.savefig()

plt.close()
pp.close()