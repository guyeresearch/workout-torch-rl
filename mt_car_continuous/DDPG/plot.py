import qn
import seaborn as sb
import pandas as pd


q_files = qn.getallfiles('log','.pkl')

data = []
for x in q_files:
    data += qn.load(x)

df = pd.DataFrame(data)
df['kind'] = 'q'


q_files = qn.getallfiles('log2','.pkl')

data = []
for x in q_files:
    data += qn.load(x)

df2 = pd.DataFrame(data)
df2['kind'] = 'q2'

df3 = pd.concat((df,df2))

sb.lineplot(x=0,y=1,hue='kind',data=df3)



