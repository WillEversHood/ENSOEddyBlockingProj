import numpy as np
import pandas as pd
import math as math
'''
Blocking events are identified as persistent events where COS and SIN eddy energy is above the 90th percentile of combined energy levels. See madeline's work for number of days required to consider it a block.
'''
'''
Criteria: The Z500 anomalies
are calculated with respect to the 1981–2010 climatology, and seasonal thresholds are set at the 90th percentile
for Z500 anomalies within the respective 3-month seasonal range (June–August, JJA; September–November,
SON; December–February, DJF; March–May, MAM). Finally, Z500 anomalies above the seasonal threshold
that persist for five or more days are accepted to indicate blocking events.
'''
'''
Citation: McKenna, M., & Karamperidou, C.
(2023). The impacts of El Niño diversity
on Northern Hemisphere atmospheric
blocking. Geophysical Research Letters,
50, e2023GL104284. https://doi.
org/10.1029/2023GL104284'''

'''
Blocking Class

Range function, input (time series), output (range of jet stream and blocking energy)

Blocking function, input (time series, time scale), Output number of blokcing events over length of time series with denominations determined by time scale'''

class Blocks():
    def __init__(self, time_series):
        self.time_series = time_series
        return

    '''
    seperate X, Y and Z, take y,z total magnitude under a 2-norm to be the eddy energy
    the 90th percentile value of that data and the 90th percentile of the jet stream energy are to be returned
    '''
    def range (self):
        self.time_series
        # make assumption that row 2 & 3 are X,Y
        max_eddy = 0
        max_jet = 0
        intermitent = 0
        '''my logic is potentially flawed here if negative values are present we'll see'''
        for i in range(self.time_series.shape[1]):
            intermitent = math.sqrt((self.time_series[1, i]**2) + (self.time_series[2, i]**2))
            if intermitent > max_value:
                max_value = intermitent
        intermitent = 0
        for i in range(self.time_series.shape[1]):
            intermitent = math.sqrt((self.time_series[0, i]**2))
            if intermitent > max_value:
                max_jet = intermitent
        eddy_threshold = max_eddy * 0.9
        jet_threshold = math_jet * 0.9
        return eddy_threshold, jet_threshold

    ''' operate based on the assumption that there is roughly a day for each day in a month we'll approximate at 30.5'''
    def blocking (self, time_scale):
        # check what time scale we are given 'month, season, year' evaluate each per year
        one = False
        two = False
        three = False
        if time_scale != 'month' or time_scale != 'season' or time_scale != 'year':
            raise ValueError('invalid timescale input')
        elif time_scale == ('month'):
            one = True
        elif time_scale == ('season'):
            two = True
        elif time_scale == ('year'):
            three = True

        # convert time series into appropriate magnitude values
        for i in range(len(self.time_series.shape[1])):
            self.time_series[1, i] = math.sqrt(self.time_series[1, i]**2 + self.time_series[2, i]**2)

        # calculate and identify blocking events
        threshold_eddy, threshold_jet = range();
        '''
        here is where the assumption that there are a sufficient number
        days per month becomes pertinent
        '''
        event = 0
        event_over = True
        for i in range(len(self.time_series.shape[1])):
            self.time_series[2, i] = 0
        # atleast 5 potentially more
        for i in range(len(self.time_series.shape[1])):
            if self.time_series[1, i] > threshold_eddy:
                event_over = False
                event += 1
            elif event_over == False:
                event_over = True
            if event >= 5 and event_over == True:
                for j in range(event - 1):
                    if j != 0:
                        self.time_series[2, i-j] = 0
        # tally and average based on number of days in month, 3 months, and 12 months respectively
        #count events and their lengths

        if one == True:
            count = 0
            #months = ['JAN', 'FEB', "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            blocks = np.empty(12, dtype=list)
            df = pd.DataFram(np.nan, index=range(100), columns = months)
            for i in range(len(self.time_serires.shape[1])):
                if self.time_series[2, i] == 1:
                    count += 1
                if self.time_series[2, i] == 0:
                    if count >= 5:
                        # add one to the pandas data frame by month and length of block
                        # little trivial symmetric group definition should produce numbers 1-12 perhaps 13 if it doesn't work quite right
                        blocks[math.floor((i%365)/30) + 1].append(count)
                    count = 0

                    

        if two == True:
            count += 1
        if three == True:
            count += 1

        return



