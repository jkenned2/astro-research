import time as t
import mysql.connector as sql
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import utils as u

# parameters
time = t.perf_counter()
sqlQueryLimit = int(0)  # limit = 0 sets there to be no limit

haEwCut = -6.
sfAssignmentRadius = .25

# obtain and array data from database
spaxelsUberData = u.sql_get('dr17_spaxels_uber',
                            'objID, r_eff_nsa, agn_flag_smc',
                            sqlQueryLimit,
                            cond=f'where spaxID in '
                                 f'(select spaxID from dr17_spaxels_elines where '
                                 f'ha_ew < {haEwCut} and '
                                 f'ha_ew > -1000)')

objIdData = spaxelsUberData[:, 0]
rEffData = spaxelsUberData[:, 1].astype(float)
flagData = spaxelsUberData[:, 2]

time = u.printActionTime(time, 'Retrieved and arrayed data from database')


# finding SF galaxies
def isSfGalaxy(objId, objIdData, rEffData, flagData):
    targetMask = np.where((objIdData == objId) & (rEffData < sfAssignmentRadius))
    targetFlags = flagData[targetMask]
    targetSfFlags = targetFlags[np.where((targetFlags == 'S03_SF') | (targetFlags == 'K03_SF'))]
    if targetSfFlags.size > 0 and targetSfFlags.size == targetFlags.size:
        return objId
    return None


uniqueObjIds = np.unique(objIdData)
vIsSfGalaxy = np.vectorize(isSfGalaxy, excluded=[1, 2, 3])
sfIds = vIsSfGalaxy(uniqueObjIds, objIdData, rEffData, flagData)
sfIds = sfIds[sfIds != None]

time = u.printActionTime(time, 'Tested galaxy types')

# writing SF galaxies to output file
u.writeNpArray('galaxy_type_ids.txt', sfIds)

time = u.printActionTime(time, 'Wrote galaxy types')
