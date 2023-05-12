import time as t
import mysql.connector as sql
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import utils as u

# parameters
time = t.perf_counter()
sqlQueryLimit = int(0)

haEwCut = -6.
haEwMax = -50.
sfAssignmentRadius = .25
rEffMax = 3.

# obtain and array data from database
sfIds = u.readNpArray('galaxy_type_ids.txt')
sfIdsSqlFormatted = u.sql_list_format(sfIds)

spaxelsUberData = u.sql_get('dr17_spaxels_uber',
                            'objID, r_eff_nsa, agn_flag_smc, agn_sn_smc, stellar_age, manga_plateifu',
                            sqlQueryLimit,
                            cond=f'where objID in {sfIdsSqlFormatted} and '
                                 f'agn_flag_smc is not null and '
                                 f'spaxID in '
                                 f'(select spaxID from dr17_spaxels_elines where ha_ew > -1000)')

spaxelsElinesData = u.sql_get('dr17_spaxels_elines',
                              'ha_ew, '
                              'log10(n2_6584_flux_corr_smc/ha_flux_corr_smc), '
                              'log10(1.33*o3_5007_flux_corr_smc/hb_flux_corr_smc), '
                              'log10((s2_6717_flux_corr_smc+s2_6731_flux_corr_smc)/ha_flux_corr_smc)',
                              sqlQueryLimit,
                              cond=f'where objID in {sfIdsSqlFormatted} and '
                                   f'ha_ew > -1000 and '
                                   f'spaxID in '
                                   f'(select spaxID from dr17_spaxels_uber where agn_flag_smc is not null)')

objIdData = spaxelsUberData[:, 0]
rEffData = spaxelsUberData[:, 1].astype(float)
flagData = spaxelsUberData[:, 2]
snData = spaxelsUberData[:, 3].astype(float)
stellarAgeData = spaxelsUberData[:, 4].astype(float)
plateIFUData = spaxelsUberData[:, 5]

haEwData = spaxelsElinesData[:, 0].astype(float)
n2HaData = spaxelsElinesData[:, 1].astype(float)
o3HbData = spaxelsElinesData[:, 2].astype(float)
s2HaData = spaxelsElinesData[:, 3].astype(float)

time = u.printActionTime(time, 'Retrieved and arrayed data from database')

# defining useful masks
agnAndRadiusMask = np.where(((flagData == 'AGN') | (flagData == 'K01_SF')) & (rEffData < rEffMax) &
                            (haEwData < haEwCut))
sfAndRadiusMask = np.where(((flagData == 'S03_SF') | (flagData == 'K03_SF')) & (rEffData < rEffMax) &
                           (haEwData < haEwCut))

# histogram of radii for types of flag
rEffSfFlags = rEffData[sfAndRadiusMask]
rEffAgnFlags = rEffData[agnAndRadiusMask]
bins = np.linspace(0., rEffMax, 75)

radiusHistFig, ax = plt.subplots()
ax.set(title=f'# Flagged Spaxels vs. radius (# AGN = {rEffAgnFlags.size}, # SF = {rEffSfFlags.size})',
       xlabel='effective radius', ylabel='# flags')
ax.hist(rEffSfFlags, bins, color='g', edgecolor='k', alpha=0.5, density=False, label='SF')
ax.hist(rEffAgnFlags, bins, color='b', edgecolor='k', alpha=0.5, density=False, label='AGN')
plt.legend()

# histogram of AGN spaxels per galaxy
agnObjIdData = objIdData[agnAndRadiusMask]
agnOrK01SfObjIds, agnOrK01SfCounts = np.unique(agnObjIdData, return_counts=True)

onlyAgnObjIdData = objIdData[np.where((flagData == 'AGN') & (rEffData < rEffMax) & (haEwData < haEwCut))]
onlyAgnObjIds, onlyAgnCounts = np.unique(onlyAgnObjIdData, return_counts=True)

agnCounts = agnOrK01SfCounts[np.isin(agnOrK01SfObjIds, onlyAgnObjIds)]

bins = np.linspace(0., np.max(agnOrK01SfCounts), 50)

agnSpaxelsHistFig, ax = plt.subplots()
ax.set(title=f'AGN spaxels in galaxies with at least one AGN spaxel ({onlyAgnObjIds.size} galaxies)',
       xlabel='# spaxels', ylabel='# galaxies')
ax.hist(agnCounts, bins, color='r', edgecolor='k', alpha=0.5, label=f'AGN and K01_SF spax (n = {np.sum(agnCounts)})')
ax.hist(onlyAgnCounts, bins, color='b', edgecolor='k', alpha=0.5, label=f'AGN spax only (n = {np.sum(onlyAgnCounts)})')
ax.legend()

# scatter plot of Ha EW vs. radius for AGN flags
haEwAgn = np.log10(-1. * haEwData[agnAndRadiusMask])
rAgn = rEffData[agnAndRadiusMask]
nBins = 75

haEwVsRadiusFig = plt.figure(figsize=(8, 5))
gs = haEwVsRadiusFig.add_gridspec(1, 2, width_ratios=(1, 6), wspace=0)

ax1 = haEwVsRadiusFig.add_subplot(gs[1])
ax1.set(title=f'H-alpha EW of AGN spaxels vs. radius (# spaxels = {haEwAgn.size})',
        xlabel='effective radius')
hb = ax1.hexbin(rAgn, haEwAgn, gridsize=nBins, bins='log')
cb = haEwVsRadiusFig.colorbar(hb, ax=ax1)
cb.set_label('log(N)')

ax2 = haEwVsRadiusFig.add_subplot(gs[0], sharey=ax1)
ax2.set(xlabel='# spaxels', ylabel='log Ha EW (log Angstroms)')
bins = np.linspace(np.min(haEwAgn), np.max(haEwAgn), nBins)
ax2.hist(haEwAgn, bins, orientation='horizontal')

# scatter plot of S/N ratio vs. radius for AGN flags
snAgn = np.log10(snData[agnAndRadiusMask])
rAgn = rEffData[agnAndRadiusMask]
nBins = 75

snVsRadiusFig = plt.figure(figsize=(8, 5))
gs = snVsRadiusFig.add_gridspec(1, 2, width_ratios=(1, 6), wspace=0)

ax1 = snVsRadiusFig.add_subplot(gs[1])
ax1.set(title=f'AGN S/N of AGN spaxels vs. radius (# spaxels = {snAgn.size})',
        xlabel='effective radius')
hb = ax1.hexbin(rAgn, snAgn, gridsize=nBins, bins='log')
cb = snVsRadiusFig.colorbar(hb, ax=ax1)
cb.set_label('log(N)')

ax2 = snVsRadiusFig.add_subplot(gs[0], sharey=ax1)
ax2.set(xlabel='# spaxels', ylabel='log AGN S/N')
bins = np.linspace(np.min(snAgn), np.max(snAgn), nBins)
ax2.hist(snAgn, bins, orientation='horizontal')

# graphing cumulative histogram of correlation coefficient for sussy SF galaxies
corrCoefs = []
pValues = []
plateIFUs = []

for objID in onlyAgnObjIds:
    mask = np.isin(objIdData, objID)
    n2Ha = n2HaData[mask]
    o3Hb = o3HbData[mask]
    plateIFU = np.unique(plateIFUData[mask])[0]
    res = stats.spearmanr(n2Ha, o3Hb)
    r, p = res.correlation, res.pvalue
    corrCoefs.append(r)
    pValues.append(p)
    plateIFUs.append(plateIFU)

corrCoefs = np.asarray(corrCoefs)
pValues = np.asarray(pValues)

rStep = 0.05
rValues = np.arange(0., 1., rStep)
nGalaxies = []
nGalaxiesLowP = []

for r in rValues:
    n = onlyAgnObjIds[np.where(corrCoefs > r)].size
    nLowP = onlyAgnObjIds[np.where((corrCoefs > r) & (pValues < 0.05))].size
    nGalaxies.append(n)
    nGalaxiesLowP.append(nLowP)

corrCoefHistFig = plt.figure(dpi=250)
gs = corrCoefHistFig.add_gridspec(2, 1, wspace=0, hspace=0)
corrCoefHistFig.suptitle('Histogram of SF Galaxies with $R_{spearman}$ > 0')

ax1 = corrCoefHistFig.add_subplot(gs[0, 0])
ax1.set(ylabel='# Galaxies with $R_{spearman} > R_0$')
ax1.step(rValues, nGalaxies, where='post', color='b', alpha=0.5, label='$p$-independent')
ax1.step(rValues, nGalaxiesLowP, where='post', color='r', alpha=0.5, label='$p < 0.05$')
ax1.legend()

ax2 = corrCoefHistFig.add_subplot(gs[1, 0], sharex=ax1)
ax2.set(xlabel='$R_0$', ylabel='# Galaxies')
ax2.hist(corrCoefs[np.where(corrCoefs > 0)], bins=rValues, color='b', alpha=0.5, label='$p$-independent')
ax2.hist(corrCoefs[np.where((corrCoefs > 0) & (pValues < 0.05))], bins=rValues, color='r', alpha=0.5,
         label='$p < 0.05$')


# bpt diagrams for AGN spaxels

# agnSort = agnCounts.argsort()
# numHighGalaxies = 10
# print(np.asarray((onlyAgnObjIds[agnSort], agnCounts[agnSort])).T[-numHighGalaxies:])

# graphing data
corrCoefThresh = 0.2
posCCGalaxies = onlyAgnObjIds[np.where((corrCoefs > corrCoefThresh) & (pValues < 0.05))]

mask = np.isin(objIdData, posCCGalaxies)
n2Ha = n2HaData[mask]
o3Hb = o3HbData[mask]
s2Ha = s2HaData[mask]
rEff = rEffData[mask]
stellarAge = stellarAgeData[mask]

nBins = 75
colorDataList = [None, rEff, stellarAge]
cbNameList = ['log N', 'eff. radius', 'stellar age (Gyr)']
cMapList = ['inferno', 'Greys', 'plasma']
binScaleList = ['log', None, None]

bptFig = plt.figure(figsize=(10, 10), dpi=250)
gs = bptFig.add_gridspec(3, 3, width_ratios=(1, 1, 0.05), wspace=0, hspace=0)
bptFig.suptitle(
    f'BPT Diagrams for Galaxies with O3 vs. N2 spearmanR > {corrCoefThresh}, p < 0.05 (# galaxies = {posCCGalaxies.size}, # spaxels = {s2Ha.size})')


class Line:
    def __init__(self, x, y, color, label):
        self.x = x
        self.y = y
        self.color = color
        self.label = label

    def maskData(self, mask):
        self.x = self.x[mask]
        self.y = self.y[mask]


bothX = np.append(n2Ha, s2Ha)
yMin, yMax = np.nanmin(o3Hb), np.nanmax(o3Hb)
diagnosticLineX = np.linspace(np.nanmin(bothX), np.nanmax(bothX), 100)

k01LineN2 = Line(diagnosticLineX, 0.61 / (diagnosticLineX - 0.47) + 1.19, 'r', 'K01')
k03LineN2 = Line(diagnosticLineX, 0.61 / (diagnosticLineX - 0.05) + 1.3, 'b', 'K03')
s06LineN2 = Line(diagnosticLineX, (-30.787 + 1.1358 * diagnosticLineX + 0.27297 * (diagnosticLineX ** 2)) * (
    np.tanh(5.7409 * diagnosticLineX)) - 31.093, 'g', 'S06')

k01LineS2 = Line(diagnosticLineX, 0.72 / (diagnosticLineX - 0.32) + 1.30, 'r', 'K01')
k06LineS2 = Line(diagnosticLineX, 1.89 * diagnosticLineX + 0.76, 'cyan', 'K06')
k06LineS2.maskData(np.where(k06LineS2.y > k01LineS2.y))

for i in range(3):
    colorData = colorDataList[i]
    cbName = cbNameList[i]
    binScale = binScaleList[i]
    cMap = cMapList[i]

    ax1 = bptFig.add_subplot(gs[i, 0])
    ax2 = bptFig.add_subplot(gs[i, 1], sharey=ax1)
    ax3 = bptFig.add_subplot(gs[i, 2])

    hexbinArr1 = ax1.hexbin(n2Ha, o3Hb, C=colorData, alpha=0, gridsize=nBins, bins=binScale).get_array()
    hexbinArr2 = ax2.hexbin(s2Ha, o3Hb, C=colorData, alpha=0, gridsize=nBins, bins=binScale).get_array()
    vMin = np.min([np.min(hexbinArr1), np.min(hexbinArr2)])
    vMax = np.max([np.max(hexbinArr1), np.max(hexbinArr2)])
    if binScale == 'log' and vMin == 0:
        vMin = 1.

    ax1.set(xlabel='log N2/Ha', ylabel='log O3/Hb')
    ax1.hexbin(n2Ha, o3Hb, C=colorData, alpha=1, gridsize=nBins, bins=binScale, cmap=cMap, vmin=vMin, vmax=vMax)

    ax2.set(xlabel='log S2/Ha')
    hb = ax2.hexbin(s2Ha, o3Hb, C=colorData, alpha=1, gridsize=nBins, bins=binScale, cmap=cMap, vmin=vMin, vmax=vMax)
    ax2.label_outer()

    cb = bptFig.colorbar(hb, cax=ax3)
    cb.set_label(cbName)

    for line in [k01LineN2, k03LineN2, s06LineN2]:
        line.maskData(np.where((line.y > yMin) & (line.y < yMax)))
        ax1.plot(line.x, line.y, color=line.color, label=line.label)
    ax1.legend()

    for line in [k01LineS2, k06LineS2]:
        line.maskData(np.where((line.y > yMin) & (line.y < yMax)))
        ax2.plot(line.x, line.y, color=line.color, label=line.label)
    ax2.legend()

# checking if sussy galaxies are in dr17_manga_mergers
sussyIDs = u.sql_list_format(posCCGalaxies)
mangaMergersData = u.sql_get('dr17_manga_mergers',
                             'objID',
                             sqlQueryLimit,
                             cond=f'where objID in {sussyIDs}')

print(np.unique(mangaMergersData))

# showing and saving figures
time = u.printActionTime(time, 'Made graphs')


class Figure:
    def __init__(self, fig, name):
        self.fig = fig
        self.name = name


Figs = [Figure(radiusHistFig, 'radius-hist'),
        Figure(haEwVsRadiusFig, 'ha-ew-vs-radius'),
        Figure(snVsRadiusFig, 'sn-vs-radius'),
        Figure(bptFig, 'bpt-diagrams'),
        Figure(agnSpaxelsHistFig, 'agn-spaxels-hist'),
        Figure(corrCoefHistFig, 'corr-coef-hist')]
FigsToShow = []
for Fig in Figs:
    if Fig.name not in FigsToShow:
        plt.close(Fig.fig)
    else:
        print(f'Showing {Fig.name}')
    Fig.fig.savefig(Fig.name)

time = u.printActionTime(time, 'Saved graphs')

plt.show()
