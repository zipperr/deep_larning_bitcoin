import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

ax = plt.subplot()
ax.plot(datetime.datetime(2019, 1, 23), 20)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set_xlim(datetime.datetime(2019, 1, 20), datetime.datetime(2019, 1, 29))
plt.show()