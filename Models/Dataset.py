import plotly.express as px
import pandas as pd
import numpy as np
from joblib import dump,load
import os
import time
class Stocks:
    def __init__(self,datadict):
        self.start_time=time.time()
        self.start=datadict["start"]
        self.end=datadict["end"]
        self.time_field=datadict["time_field"]
        self.code_field=datadict["code_field"]
        self.save_path=datadict["save_path"]
        self.Name2Columns=datadict["name_dict"]

        if datadict["load"]:
            if not self.load_from_joblib(self.save_path):
                self.Stock_dict= self.StkProcessing(datadict["files"], self.Name2Columns,self.time_field, self.code_field)
                self.save_to_joblib(self.save_path)
        else:
            self.Stock_dict= self.StkProcessing(datadict["files"], self.Name2Columns,self.time_field, self.code_field)
        self.Oprc=self.Stock_dict['Oprc']
        self.Clprc=self.Stock_dict['Clprc']
        self.Tnovr=self.Stock_dict['Tnovr']
        self.MV=self.Stock_dict['MV']
        self.CMV=self.Stock_dict['CMV']
       



    def StkProcessing(self,files,Name2Columns,time_field,code_field):
        df= pd.concat([pd.read_csv(file) for file in files])
        df=df[df['Filling']==0]
        df_dict={}
        for name,column in Name2Columns.items():
            col_df = df[df[code_field].notnull()][[time_field,code_field, column]]
            col_df = col_df.pivot(index=time_field, columns=code_field, values=column)
            df_dict[name] = col_df
        print(f"Stock_dict generated in {time.time()-self.start_time} seconds")
        return df_dict
    
    def save_to_joblib(self, file_path):
        dump(self.Stock_dict, file_path)   
        print(f"Stock_dict saved to {file_path}")
    
    def load_from_joblib(self, file_path):
        if os.path.exists(file_path):
            self.Stock_dict = load(file_path)
            print(f"Stock_dict loaded from {file_path} in {time.time()-self.start_time} seconds")
            return True
        else:
            print(f"{file_path} not found, redirecting to generator.")
            return False
        
    def Interplolate(self,df,time,order=2):
        df.loc[:,time]=pd.to_datetime(df[time])
        df.set_index(time,inplace=True)
        date_range = pd.date_range(start=self.start, end=self.end, freq='D')
        df = df[self.start:self.end].reindex(date_range)
        if np.isnan(df.iloc[0].item()):
            df.iloc[0]=df.mean().item()
        return df,df.interpolate(method='spline',order=order)

    def BfAfInterplolate(self,df,df_filled):
        fig = px.line(x=df.index, y=df[df.columns[0]], line_shape='linear', title="Yield over Time")

        fig.add_scatter(x=df_filled.index, y=df_filled[df_filled.columns[0]], mode='lines', name="Interpolated Data", line=dict(color='red'))
        var=np.var(df_filled.to_numpy())
        mean=np.mean(df_filled.to_numpy())
        fig.update_traces(connectgaps=False)
        fig.update_layout(
            title="Yield over Time",
            xaxis_title="Date",
            yaxis_title="Yield(%)",
            legend_title="Data",
            legend=dict(
                title="Legend",
                x=1,
                y=1
            )
        )
    
        fig.add_annotation(
            x=df_filled.index[len(df_filled)//2],  # 放置注释的x位置
            y=max(df_filled[df_filled.columns[0]]),  # 放置注释的y位置
            text=f"Variance: {var:.4f}Mean: {mean:.4f}",  # 方差值，保留四位小数
            showarrow=False,  # 不显示箭头
            font=dict(size=12, color='black'),  # 设置字体大小和颜色
            bgcolor='white'  # 背景颜色
        )
    
        fig.show()

    def CalYield(self,Clsprc):
        e=Clsprc.to_numpy()
        e_filled = np.zeros(e.shape[0] + 1) 
        e_filled[1:]=e
        e_filled[0]=2*e[0]-e[1]
        Yield=np.log(e_filled[1:]/e_filled[:-1])*100
        return Yield