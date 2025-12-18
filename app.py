# -*- coding: utf-8 -*-
import streamlit as st
import json, re, pandas as pd, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patches as patches
from io import BytesIO
from mplsoccer import Pitch
import matplotlib.patheffects as path_effects
from unidecode import unidecode
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Zona del Analista", page_icon="⚽", layout="wide")
violet = '#a369ff'

def extract_json_from_html(html_content):
    regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
    data_txt = re.findall(regex_pattern, html_content)[0]
    for old, new in [('matchId', '"matchId"'), ('matchCentreData', '"matchCentreData"'), 
                     ('matchCentreEventTypeJson', '"matchCentreEventTypeJson"'), 
                     ('formationIdNameMappings', '"formationIdNameMappings"'), ('};', '}')]:
        data_txt = data_txt.replace(old, new)
    return data_txt

def extract_data_from_dict(data):
    events_dict = data["matchCentreData"]["events"]
    teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                  data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
    players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
    players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
    players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
    players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    return events_dict, players_df, teams_dict

def process_dataframe(df, teams_dict):
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else x)
    period_map = {'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5}
    df['period'] = df['period'].map(lambda x: period_map.get(x, x) if isinstance(x, str) else x)
    df['period'] = pd.to_numeric(df['period'], errors='coerce').fillna(1).astype(int)
    df['teamName'] = df['teamId'].map(teams_dict)
    df['x'] = pd.to_numeric(df['x'], errors='coerce') * 1.05
    df['y'] = pd.to_numeric(df['y'], errors='coerce') * 0.68
    df['endX'] = pd.to_numeric(df.get('endX'), errors='coerce') * 1.05 if 'endX' in df.columns else np.nan
    df['endY'] = pd.to_numeric(df.get('endY'), errors='coerce') * 0.68 if 'endY' in df.columns else np.nan
    if 'qualifiers' in df.columns: df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'), np.sqrt((105-df['x'])**2+(34-df['y'])**2)-np.sqrt((105-df['endX'])**2+(34-df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'), np.sqrt((105-df['x'])**2+(34-df['y'])**2)-np.sqrt((105-df['endX'])**2+(34-df['endY'])**2), 0)
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
    df['second'] = pd.to_numeric(df['second'], errors='coerce').fillna(0)
    df['cumulative_mins'] = df['minute'] + df['second']/60
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY']-df['y'], df['endX']-df['x']))
    return df

def insert_ball_carries(events_df):
    match_events = events_df.reset_index(drop=True)
    carries = []
    for idx in range(len(match_events)-1):
        ev, nxt = match_events.loc[idx], match_events.loc[idx+1]
        if pd.isna(ev.get('endX')): continue
        dx, dy = nxt['x']-ev['endX'], nxt['y']-ev['endY']
        if ev['teamId']==nxt['teamId'] and nxt['type'] not in ['BallTouch','TakeOn','Foul'] and 9<=dx**2+dy**2<=3600:
            carries.append({'minute':ev['minute'],'second':ev['second'],'teamId':nxt['teamId'],'teamName':nxt['teamName'],
                'x':ev['endX'],'y':ev['endY'],'endX':nxt['x'],'endY':nxt['y'],'type':'Carry','outcomeType':'Successful',
                'period':nxt['period'],'playerId':nxt.get('playerId'),'name':nxt.get('name'),'position':nxt.get('position'),
                'cumulative_mins':(ev['cumulative_mins']+nxt['cumulative_mins'])/2,
                'prog_carry':np.sqrt((105-ev['endX'])**2+(34-ev['endY'])**2)-np.sqrt((105-nxt['x'])**2+(34-nxt['y'])**2)})
    return pd.concat([pd.DataFrame(carries), match_events]).sort_values(['period','cumulative_mins']).reset_index(drop=True)

def get_short_name(n): 
    if not isinstance(n,str): return str(n) if n else ''
    p=n.split(); return n if len(p)==1 else p[0][0]+". "+p[-1]

def get_passes_df(df):
    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card', na=False)].copy()
    df1["receiver"] = df1["playerId"].shift(-1)
    return df1.loc[df1['type']=='Pass', ["x","y","endX","endY","teamName","playerId","receiver","type","outcomeType","pass_or_carry_angle"]]

def get_passes_between_df(teamName, passes_df, df, players_df):
    passes_df = passes_df[passes_df["teamName"]==teamName].copy()
    dfteam = df[(df['teamName']==teamName)&(~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card',na=False))]
    passes_df = passes_df.merge(players_df[["playerId","isFirstEleven"]], on='playerId', how='left')
    avg_locs = dfteam.groupby('playerId').agg({'x':['median'],'y':['median','count']})
    avg_locs.columns = ['pass_avg_x','pass_avg_y','count']
    avg_locs = avg_locs.merge(players_df[['playerId','name','shirtNo','position','isFirstEleven']], on='playerId', how='left').set_index('playerId')
    passes_ids = passes_df[['playerId','receiver','teamName']].copy()
    passes_ids['pos_max'] = passes_ids[['playerId','receiver']].max(axis=1)
    passes_ids['pos_min'] = passes_ids[['playerId','receiver']].min(axis=1)
    passes_between = passes_ids.groupby(['pos_min','pos_max']).size().reset_index(name='pass_count')
    passes_between = passes_between.merge(avg_locs[['pass_avg_x','pass_avg_y','name']], left_on='pos_min', right_index=True)
    passes_between = passes_between.merge(avg_locs[['pass_avg_x','pass_avg_y','name']], left_on='pos_max', right_index=True, suffixes=['','_end'])
    return passes_between, avg_locs

def get_defensive_action_df(df):
    ids = df.index[((df['type']=='Aerial')&(df['qualifiers'].str.contains('Defensive',na=False)))|(df['type'].isin(['BallRecovery','BlockedPass','Challenge','Clearance','Foul','Interception','Tackle']))]
    return df.loc[ids, ["x","y","teamName","playerId","type","outcomeType"]]

def get_da_count_df(team_name, da_df, players_df):
    da_df = da_df[da_df["teamName"]==team_name].merge(players_df[["playerId","isFirstEleven"]], on='playerId', how='left')
    avg = da_df.groupby('playerId').agg({'x':['median'],'y':['median','count']})
    avg.columns = ['x','y','count']
    return avg.merge(players_df[['playerId','name','shirtNo','position','isFirstEleven']], on='playerId', how='left').set_index('playerId')

def create_progressor_df(df, team_name):
    team_df = df[df['teamName']==team_name]
    players = team_df['name'].dropna().unique()
    data = {'name':[],'Progressive Passes':[],'Progressive Carries':[],'LineBreaking Pass':[]}
    for n in players:
        data['name'].append(n)
        data['Progressive Passes'].append(len(df[(df['name']==n)&(df['prog_pass']>=9.144)&(df['x']>=35)&(df['outcomeType']=='Successful')&(~df['qualifiers'].str.contains('CornerTaken|Freekick',na=False))]))
        data['Progressive Carries'].append(len(df[(df['name']==n)&(df['prog_carry']>=9.144)&(df['endX']>=35)]))
        data['LineBreaking Pass'].append(len(df[(df['name']==n)&(df['type']=='Pass')&(df['qualifiers'].str.contains('Throughball',na=False))]))
    pf = pd.DataFrame(data)
    pf['total'] = pf['Progressive Passes']+pf['Progressive Carries']+pf['LineBreaking Pass']
    pf = pf.sort_values('total',ascending=False).reset_index(drop=True)
    pf['shortName'] = pf['name'].apply(get_short_name)
    return pf

def create_defender_df(df, team_name):
    team_df = df[df['teamName']==team_name]
    players = team_df['name'].dropna().unique()
    data = {'name':[],'Tackles':[],'Interceptions':[],'Clearance':[]}
    for n in players:
        data['name'].append(n)
        data['Tackles'].append(len(df[(df['name']==n)&(df['type']=='Tackle')&(df['outcomeType']=='Successful')]))
        data['Interceptions'].append(len(df[(df['name']==n)&(df['type']=='Interception')]))
        data['Clearance'].append(len(df[(df['name']==n)&(df['type']=='Clearance')]))
    df2 = pd.DataFrame(data)
    df2['total'] = df2['Tackles']+df2['Interceptions']+df2['Clearance']
    df2 = df2.sort_values('total',ascending=False).reset_index(drop=True)
    df2['shortName'] = df2['name'].apply(get_short_name)
    return df2

def create_shot_sequence_df(df, team_name):
    team_df = df[df['teamName']==team_name]
    players = team_df['name'].dropna().unique()
    df_nc = df[df['type']!='Carry']
    data = {'name':[],'Shots':[],'Shot Assist':[],'Buildup to shot':[]}
    for n in players:
        data['name'].append(n)
        data['Shots'].append(len(df[(df['name']==n)&(df['type'].isin(['MissedShots','SavedShot','ShotOnPost','Goal']))]))
        data['Shot Assist'].append(len(df[(df['name']==n)&(df['type']=='Pass')&(df['qualifiers'].str.contains('KeyPass',na=False))]))
        data['Buildup to shot'].append(len(df_nc[(df_nc['name']==n)&(df_nc['type']=='Pass')&(df_nc['qualifiers'].shift(-1).str.contains('KeyPass',na=False))]))
    df2 = pd.DataFrame(data)
    df2['total'] = df2['Shots']+df2['Shot Assist']+df2['Buildup to shot']
    df2 = df2.sort_values('total',ascending=False).reset_index(drop=True)
    df2['shortName'] = df2['name'].apply(get_short_name)
    return df2

# === VISUALIZACIONES DASHBOARD 1 ===
def pass_network_visualization(ax, passes_between_df, avg_locs, col, teamName, hteamName, bg_color, line_color, passes_df):
    passes_between_df = passes_between_df.copy()
    passes_between_df['width'] = passes_between_df.pass_count/passes_between_df.pass_count.max()*15
    color = np.tile(np.array(to_rgba(col)), (len(passes_between_df),1))
    color[:,3] = (passes_between_df.pass_count/passes_between_df.pass_count.max()*0.8)+0.1
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end, lw=passes_between_df.width, color=color, zorder=1, ax=ax)
    for _,row in avg_locs.iterrows():
        m = 'o' if row.get('isFirstEleven',True) else 's'
        pitch.scatter(row['pass_avg_x'],row['pass_avg_y'],s=1000,marker=m,color=bg_color,edgecolor=line_color,linewidth=2,ax=ax)
        pitch.annotate(str(int(row["shirtNo"])) if pd.notna(row.get("shirtNo")) else '',xy=(row.pass_avg_x,row.pass_avg_y),c=col,ha='center',va='center',size=18,ax=ax)
    avgph = round(avg_locs['pass_avg_x'].median(),2)
    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)
    cbs = avg_locs[avg_locs['position']=='DC']
    def_h = round(cbs['pass_avg_x'].median(),2) if not cbs.empty else avgph
    fwds = avg_locs[avg_locs['isFirstEleven']==1].nlargest(2,'pass_avg_x')
    fwd_h = round(fwds['pass_avg_x'].mean(),2) if not fwds.empty else avgph
    ax.fill([def_h,fwd_h,fwd_h,def_h],[0,0,68,68],col,alpha=0.1)
    tp = passes_df[passes_df["teamName"]==teamName].copy()
    tp['pass_or_carry_angle'] = tp['pass_or_carry_angle'].abs()
    tp = tp[(tp['pass_or_carry_angle']>=0)&(tp['pass_or_carry_angle']<=90)]
    vert = round((1-tp['pass_or_carry_angle'].median()/90)*100,2) if not tp.empty else 0
    if teamName!=hteamName:
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.text(avgph-1,73,f"{avgph}m",fontsize=15,color=line_color,ha='left')
        ax.text(105,73,f"verticalidad: {vert}%",fontsize=15,color=line_color,ha='left')
        ax.text(2,2,"circulo = Titulares\ncuadro= suplentes",color=col,size=12,ha='right',va='top')
        ax.set_title(f"{teamName}\nRed de pases",color=line_color,size=25,fontweight='bold')
    else:
        ax.text(avgph-1,-5,f"{avgph}m",fontsize=15,color=line_color,ha='right')
        ax.text(105,-5,f"verticalidad: {vert}%",fontsize=15,color=line_color,ha='right')
        ax.text(2,66,"circulo = Titulares\ncuadro= suplentes",color=col,size=12,ha='left',va='top')
        ax.set_title(f"{teamName}\nRed de Pases",color=line_color,size=25,fontweight='bold')

def plot_shotmap(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
    pitch.draw(ax=ax); ax.set_ylim(-0.5,68.5); ax.set_xlim(-0.5,105.5)
    shots = df[df['type'].isin(['Goal','MissedShots','SavedShot','ShotOnPost'])]
    for team,col,flip in [(hteamName,hcol,True),(ateamName,acol,False)]:
        s = shots[shots['teamName']==team]
        g,p,sv,m = s[s['type']=='Goal'],s[s['type']=='ShotOnPost'],s[s['type']=='SavedShot'],s[s['type']=='MissedShots']
        if flip:
            pitch.scatter(105-g.x,68-g.y,s=350,edgecolors='white',c='None',marker='football',zorder=3,ax=ax)
            pitch.scatter(105-p.x,68-p.y,s=200,edgecolors=col,c=col,marker='o',ax=ax)
            pitch.scatter(105-sv.x,68-sv.y,s=200,edgecolors=col,c='None',hatch='///////',marker='o',ax=ax)
            pitch.scatter(105-m.x,68-m.y,s=200,edgecolors=col,c='None',marker='o',ax=ax)
        else:
            pitch.scatter(g.x,g.y,s=350,edgecolors='white',c='None',marker='football',zorder=3,ax=ax)
            pitch.scatter(p.x,p.y,s=200,edgecolors=col,c=col,marker='o',ax=ax)
            pitch.scatter(sv.x,sv.y,s=200,edgecolors=col,c='None',hatch='///////',marker='o',ax=ax)
            pitch.scatter(m.x,m.y,s=200,edgecolors=col,c='None',marker='o',ax=ax)
    ax.text(0,70,f"{hteamName}\n<---Tiros",color=hcol,size=25,ha='left',fontweight='bold')
    ax.text(105,70,f"{ateamName}\nTiros--->",color=acol,size=25,ha='right',fontweight='bold')

def defensive_block(ax, avg_locs, da_df, team_name, col, hteamName, bg_color, line_color):
    da_team = da_df[da_df["teamName"]==team_name]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_facecolor(bg_color); ax.set_xlim(-0.5,105.5)
    avg_locs = avg_locs.copy()
    avg_locs['marker_size'] = avg_locs['count']/avg_locs['count'].max()*3500
    cm = LinearSegmentedColormap.from_list("c",[bg_color,col],N=500)
    if len(da_team)>2:
        try: pitch.kdeplot(da_team.x,da_team.y,ax=ax,fill=True,levels=5000,thresh=0.02,cut=4,cmap=cm)
        except: pass
    for _,row in avg_locs.iterrows():
        m = 'o' if row.get('isFirstEleven',True) else 's'
        pitch.scatter(row['x'],row['y'],s=row['marker_size']+100,marker=m,color=bg_color,edgecolor=line_color,linewidth=1,zorder=3,ax=ax)
        pitch.annotate(str(int(row["shirtNo"])) if pd.notna(row.get("shirtNo")) else '',xy=(row.x,row.y),c=line_color,ha='center',va='center',size=14,ax=ax)
    dah = round(avg_locs['x'].mean(),2)
    ax.axvline(x=dah,color='gray',linestyle='--',alpha=0.75,linewidth=2)
    cbs = avg_locs[avg_locs['position']=='DC']
    def_h = round(cbs['x'].median(),2) if not cbs.empty else dah
    fwds = avg_locs[avg_locs['isFirstEleven']==1].nlargest(2,'x')
    fwd_h = round(fwds['x'].mean(),2) if not fwds.empty else dah
    comp = round((1-((fwd_h-def_h)/105))*100,2) if fwd_h!=def_h else 0
    if team_name!=hteamName:
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.text(dah-1,73,f"{dah}m",fontsize=15,color=line_color,ha='left')
        ax.text(105,73,f'Defensa Compacta: {comp}%',fontsize=15,color=line_color,ha='left')
        ax.text(2,2,"círculo = titular\ncuadro = suplente",color='gray',size=12,ha='right',va='top')
        ax.set_title(f"{team_name}\nBloque Defensivo",color=line_color,fontsize=25,fontweight='bold')
    else:
        ax.text(dah-1,-5,f"{dah}m",fontsize=15,color=line_color,ha='right')
        ax.text(105,-5,f'Defensa Compacta: {comp}%',fontsize=15,color=line_color,ha='right')
        ax.text(2,66,"círculo = titular\ncuadro = suplente",color='gray',size=12,ha='left',va='top')
        ax.set_title(f"{team_name}\nBloque Defensivo",color=line_color,fontsize=25,fontweight='bold')

def plot_goalPost(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=bg_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_ylim(-0.5,68.5); ax.set_xlim(-0.5,105.5)
    for y in [7.5,97.5]: ax.plot([y,y],[0,30],color=line_color,linewidth=5)
    ax.plot([7.5,97.5],[30,30],color=line_color,linewidth=5)
    ax.plot([0,105],[0,0],color=line_color,linewidth=3)
    for y in [7.5,97.5]: ax.plot([y,y],[38,68],color=line_color,linewidth=5)
    ax.plot([7.5,97.5],[68,68],color=line_color,linewidth=5)
    ax.plot([0,105],[38,38],color=line_color,linewidth=3)
    shots = df[df['type'].isin(['Goal','MissedShots','SavedShot','ShotOnPost'])]
    hs = len(shots[(shots['teamName']==hteamName)&(shots['type']=='SavedShot')])
    avs = len(shots[(shots['teamName']==ateamName)&(shots['type']=='SavedShot')])
    ax.text(52.5,70,f"{hteamName} PT Paradas",color=hcol,fontsize=25,ha='center',fontweight='bold')
    ax.text(52.5,-2,f"{ateamName} PT Paradas",color=acol,fontsize=25,ha='center',va='top',fontweight='bold')
    ax.text(100,68,f"Paradas = {avs}",color=hcol,fontsize=14,va='top',ha='left')
    ax.text(100,2,f"Paradas = {hs}",color=acol,fontsize=14,va='bottom',ha='left')

def draw_progressive_pass_map(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpro = df[(df['teamName']==team_name)&(df['prog_pass']>=9.11)&(~df['qualifiers'].str.contains('CornerTaken|Freekick',na=False))&(df['x']>=35)&(df['outcomeType']=='Successful')]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    if team_name!=hteamName: ax.invert_xaxis(); ax.invert_yaxis()
    pc = len(dfpro)
    if pc==0: ax.set_title(f"{team_name}\n0 Pases Progresivos",color=line_color,fontsize=25,fontweight='bold'); return
    l,m,r = len(dfpro[dfpro['y']>=45.33]),len(dfpro[(dfpro['y']>=22.67)&(dfpro['y']<45.33)]),len(dfpro[dfpro['y']<22.67])
    ax.hlines([22.67,45.33],0,105,colors=line_color,linestyle='dashed',alpha=0.35)
    bb = dict(boxstyle="round,pad=0.3",edgecolor="None",facecolor=bg_color,alpha=0.75)
    ax.text(8,11.335,f'{r}\n({round(r/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,34,f'{m}\n({round(m/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,56.675,f'{l}\n({round(l/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    pitch.lines(dfpro.x,dfpro.y,dfpro.endX,dfpro.endY,lw=3.5,comet=True,color=col,ax=ax,alpha=0.5)
    pitch.scatter(dfpro.endX,dfpro.endY,s=35,edgecolor=col,linewidth=1,color=bg_color,zorder=2,ax=ax)
    ax.set_title(f"{team_name}\n{pc} Pases Progresivos",color=line_color,fontsize=25,fontweight='bold')

def draw_progressive_carry_map(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpro = df[(df['teamName']==team_name)&(df['type']=='Carry')&(df['prog_carry']>=9.11)&(df['endX']>=35)]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    if team_name!=hteamName: ax.invert_xaxis(); ax.invert_yaxis()
    pc = len(dfpro)
    if pc==0: ax.set_title(f"{team_name}\n0 Avance Progresivo con Balón",color=line_color,fontsize=25,fontweight='bold'); return
    l,m,r = len(dfpro[dfpro['y']>=45.33]),len(dfpro[(dfpro['y']>=22.67)&(dfpro['y']<45.33)]),len(dfpro[dfpro['y']<22.67])
    ax.hlines([22.67,45.33],0,105,colors=line_color,linestyle='dashed',alpha=0.35)
    bb = dict(boxstyle="round,pad=0.3",edgecolor="None",facecolor=bg_color,alpha=0.75)
    ax.text(8,11.335,f'{r}\n({round(r/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,34,f'{m}\n({round(m/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,56.675,f'{l}\n({round(l/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    for _,row in dfpro.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            ax.add_patch(patches.FancyArrowPatch((row['x'],row['y']),(row['endX'],row['endY']),arrowstyle='->',color=col,zorder=4,mutation_scale=20,alpha=0.9,linewidth=2,linestyle='--'))
    ax.set_title(f"{team_name}\n{pc} Avance Progresivo con Balón",color=line_color,fontsize=25,fontweight='bold')

def plotting_match_stats(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    ax.set_facecolor(bg_color); ax.set_xlim(0,105); ax.set_ylim(-5,65)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    def gs(team):
        t = df[df['teamName']==team]
        p = t[t['type']=='Pass']; a = p[p['outcomeType']=='Successful']
        lb = p[p['qualifiers'].str.contains('Longball',na=False)]; lba = lb[lb['outcomeType']=='Successful']
        co = p[p['qualifiers'].str.contains('CornerTaken',na=False)]
        tk = t[t['type']=='Tackle']; tkw = tk[tk['outcomeType']=='Successful']
        i = t[t['type']=='Interception']; c = t[t['type']=='Clearance']
        ae = t[t['type']=='Aerial']; aew = ae[ae['outcomeType']=='Successful']
        return len(p),len(a),len(lb),len(lba),len(co),len(tk),len(tkw),len(i),len(c),len(ae),len(aew)
    hp,hac,hlb,hlba,hcor,htk,htkw,hint,hcl,har,harw = gs(hteamName)
    ap,aac,alb,alba,acor,atk,atkw,aint,acl,aar,aarw = gs(ateamName)
    tot = hp+ap; hposs = round(hp/tot*100) if tot>0 else 50; aposs = 100-hposs
    pe = [path_effects.Stroke(linewidth=3,foreground=bg_color),path_effects.Normal()]
    stats = ['Posesion','Pases (Acc.)','Balones Largos (Acc.)','Corners','Entradas (ganadas)','Intercepciones','Despejes','Duelos aéreos (ganados)']
    hv = [f"{hposs}%",f"{hp}({hac})",f"{hlb}({hlba})",str(hcor),f"{htk}({htkw})",str(hint),str(hcl),f"{har}({harw})"]
    av = [f"{aposs}%",f"{ap}({aac})",f"{alb}({alba})",str(acor),f"{atk}({atkw})",str(aint),str(acl),f"{aar}({aarw})"]
    ax.set_title("Estadisticas Partido",color=line_color,fontsize=25,fontweight='bold')
    for i,(s,h,a) in enumerate(zip(stats,hv,av)):
        y = 55-i*7
        ax.text(52.5,y,s,color=bg_color,fontsize=15,ha='center',va='center',fontweight='bold',path_effects=pe)
        ax.text(5,y,h,color=line_color,fontsize=17,ha='left',va='center',fontweight='bold')
        ax.text(100,y,a,color=line_color,fontsize=17,ha='right',va='center',fontweight='bold')

# === VISUALIZACIONES DASHBOARD 2 ===
def Final_third_entry(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfpass = df[(df['teamName']==team_name)&(df['type']=='Pass')&(df['x']<70)&(df['endX']>=70)&(df['outcomeType']=='Successful')&(~df['qualifiers'].str.contains('Freekick',na=False))]
    dfcarry = df[(df['teamName']==team_name)&(df['type']=='Carry')&(df['x']<70)&(df['endX']>=70)]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    if team_name!=hteamName: ax.invert_xaxis(); ax.invert_yaxis()
    pc = len(dfpass)+len(dfcarry)
    if pc==0: ax.set_title(f"{team_name}\n0 Entradas al último tercio",color=line_color,fontsize=25,fontweight='bold'); return
    l = len(dfpass[dfpass['y']>=45.33])+len(dfcarry[dfcarry['y']>=45.33])
    m = len(dfpass[(dfpass['y']>=22.67)&(dfpass['y']<45.33)])+len(dfcarry[(dfcarry['y']>=22.67)&(dfcarry['y']<45.33)])
    r = len(dfpass[dfpass['y']<22.67])+len(dfcarry[dfcarry['y']<22.67])
    ax.hlines([22.67,45.33],0,70,colors=line_color,linestyle='dashed',alpha=0.35)
    ax.vlines(70,-2,70,colors=line_color,linestyle='dashed',alpha=0.55)
    bb = dict(boxstyle="round,pad=0.3",edgecolor="None",facecolor=bg_color,alpha=0.75)
    ax.text(8,11.335,f'{r}\n({round(r/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,34,f'{m}\n({round(m/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    ax.text(8,56.675,f'{l}\n({round(l/pc*100)}%)',color=col,fontsize=24,va='center',ha='center',bbox=bb)
    pitch.lines(dfpass.x,dfpass.y,dfpass.endX,dfpass.endY,lw=2,comet=True,color=col,ax=ax,alpha=0.5)
    for _,row in dfcarry.iterrows():
        if pd.notna(row['x']) and pd.notna(row['endX']):
            ax.add_patch(patches.FancyArrowPatch((row['x'],row['y']),(row['endX'],row['endY']),arrowstyle='->',color=col,zorder=4,mutation_scale=15,alpha=0.7,linewidth=1.5,linestyle='--'))
    ax.set_title(f"{team_name}\n{pc} Entradas al último tercio",color=line_color,fontsize=25,fontweight='bold')

def box_entry(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    def ge(team):
        p = df[(df['teamName']==team)&(df['type']=='Pass')&(df['outcomeType']=='Successful')&(df['x']<88.5)&(df['endX']>=88.5)&(df['endY']>=13.85)&(df['endY']<=54.15)]
        c = df[(df['teamName']==team)&(df['type']=='Carry')&(df['x']<88.5)&(df['endX']>=88.5)&(df['endY']>=13.85)&(df['endY']<=54.15)]
        lp = len(p[p['y']>=34])+len(c[c['y']>=34]); rp = len(p[p['y']<34])+len(c[c['y']<34])
        return len(p)+len(c),lp,rp
    ht,hl,hr = ge(hteamName); at,al,ar = ge(ateamName)
    bb = lambda c: dict(boxstyle="round,pad=0.5",facecolor=bg_color,edgecolor=c,linewidth=2)
    ax.text(30,50,f"{hl}",color=hcol,fontsize=30,ha='center',va='center',fontweight='bold',bbox=bb(hcol))
    ax.text(30,18,f"{hr}",color=hcol,fontsize=30,ha='center',va='center',fontweight='bold',bbox=bb(hcol))
    ax.text(75,50,f"{al}",color=acol,fontsize=30,ha='center',va='center',fontweight='bold',bbox=bb(acol))
    ax.text(75,18,f"{ar}",color=acol,fontsize=30,ha='center',va='center',fontweight='bold',bbox=bb(acol))
    ax.set_title(f"{hteamName}\nEntradas al área: {ht}",color=hcol,fontsize=20,fontweight='bold',loc='left')
    ax.set_title(f"{ateamName}\nEntradas al área: {at}",color=acol,fontsize=20,fontweight='bold',loc='right')

def zone14hs(ax, df, team_name, col, hteamName, bg_color, line_color):
    dfhp = df[(df['teamName']==team_name)&(df['type']=='Pass')&(df['outcomeType']=='Successful')&(~df['qualifiers'].str.contains('CornerTaken|Freekick',na=False))]
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    if team_name!=hteamName: ax.invert_xaxis(); ax.invert_yaxis()
    z14 = hs = 0
    for _,row in dfhp.iterrows():
        if row['endX']>=70 and row['endX']<=88.54 and row['endY']>=22.66 and row['endY']<=45.32:
            pitch.lines(row['x'],row['y'],row['endX'],row['endY'],color='#38dacc',comet=True,lw=3,zorder=3,ax=ax,alpha=0.75)
            ax.scatter(row['endX'],row['endY'],s=35,linewidth=1,color=bg_color,edgecolor='#38dacc',zorder=4)
            z14 += 1
        if row['endX']>=70 and ((row['endY']>=11.33 and row['endY']<=22.66) or (row['endY']>=45.32 and row['endY']<=56.95)):
            pitch.lines(row['x'],row['y'],row['endX'],row['endY'],color=col,comet=True,lw=3,zorder=3,ax=ax,alpha=0.75)
            ax.scatter(row['endX'],row['endY'],s=35,linewidth=1,color=bg_color,edgecolor=col,zorder=4)
            hs += 1
    ax.fill([70,88.54,88.54,70],[22.66,22.66,45.32,45.32],'#38dacc',alpha=0.2)
    ax.fill([70,105,105,70],[11.33,11.33,22.66,22.66],col,alpha=0.2)
    ax.fill([70,105,105,70],[45.32,45.32,56.95,56.95],col,alpha=0.2)
    ax.scatter(16.46,13.85,color=col,s=10000,edgecolor=line_color,linewidth=2,marker='h')
    ax.scatter(16.46,54.15,color='#38dacc',s=10000,edgecolor=line_color,linewidth=2,marker='h')
    ax.text(16.46,13.85-3.5,"Carril int",fontsize=16,color=line_color,ha='center')
    ax.text(16.46,54.15-3.5,"Zona14",fontsize=16,color=line_color,ha='center')
    ax.text(16.46,13.85+2,str(hs),fontsize=35,color=line_color,ha='center',va='center',fontweight='bold')
    ax.text(16.46,54.15+2,str(z14),fontsize=35,color=line_color,ha='center',va='center',fontweight='bold')
    ax.set_title(f"{team_name}\nPase Zona 14 y carril interior",color=line_color,fontsize=25,fontweight='bold')

def Crosses(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    def gc(team):
        c = df[(df['teamName']==team)&(df['type']=='Pass')&(df['qualifiers'].str.contains('Cross',na=False))&(~df['qualifiers'].str.contains('CornerTaken',na=False))]
        s = len(c[c['outcomeType']=='Successful']); l = len(c[c['y']>=45.33]); r = len(c[c['y']<22.67])
        return len(c),s,l,r
    ht,hs,hl,hr = gc(hteamName); at,asuc,al,ar = gc(ateamName)
    ax.text(20,55,f"Centros desde\nla Derecha: {hr}",color=hcol,fontsize=14,ha='center')
    ax.text(20,13,f"Centros desde\nla Izquierda: {hl}",color=hcol,fontsize=14,ha='center')
    ax.text(85,55,f"Centros desde\nla Izquierda: {al}",color=acol,fontsize=14,ha='center')
    ax.text(85,13,f"Centros desde\nLa Derecha: {ar}",color=acol,fontsize=14,ha='center')
    ax.set_title(f"{hteamName}\n<---Centros",color=hcol,fontsize=20,fontweight='bold',loc='left')
    ax.set_title(f"{ateamName}\nCentros--->",color=acol,fontsize=20,fontweight='bold',loc='right')
    ax.text(20,-3,f"Acertados: {hs}\nFallados: {ht-hs}",color=hcol,fontsize=12,ha='center')
    ax.text(85,-3,f"Acertados: {asuc}\nFallados: {at-asuc}",color=acol,fontsize=12,ha='center')

def Pass_end_zone(ax, df, team_name, col, hteamName, bg_color, line_color):
    pez = df[(df['teamName']==team_name)&(df['type']=='Pass')&(df['outcomeType']=='Successful')]
    pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    if team_name!=hteamName: ax.invert_xaxis(); ax.invert_yaxis()
    cm = LinearSegmentedColormap.from_list("c",[bg_color,col],N=20)
    if not pez.empty:
        bs = pitch.bin_statistic(pez['endX'],pez['endY'],statistic='count',bins=(6,5))
        pitch.heatmap(bs,ax=ax,cmap=cm,edgecolors=line_color,linewidth=0.5,alpha=0.7)
        bs['statistic'] = (bs['statistic']/bs['statistic'].sum()*100).round(0).astype(int)
        pitch.label_heatmap(bs,ax=ax,str_format='{:.0f}%',color=line_color,fontsize=12,va='center',ha='center')
    ax.set_title(f"{team_name}\nZona de finalización pase",color=line_color,fontsize=25,fontweight='bold')

def HighTO(ax, df, hteamName, ateamName, hcol, acol, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5)
    def gt(team):
        t = df[df['teamName']==team]
        to = t[(t['type']=='Dispossessed')|((t['type']=='Pass')&(t['outcomeType']=='Unsuccessful'))]
        return len(to[to['x']>=70])
    hto,ato = gt(hteamName),gt(ateamName)
    ax.fill([0,35,35,0],[0,0,68,68],hcol,alpha=0.15)
    ax.fill([70,105,105,70],[0,0,68,68],acol,alpha=0.15)
    ax.set_title(f"{hteamName}\nPérdidas zona alta: {hto}",color=hcol,fontsize=18,fontweight='bold',loc='left')
    ax.set_title(f"{ateamName}\nPérdidas zona alta: {ato}",color=acol,fontsize=18,fontweight='bold',loc='right')

# === VISUALIZACIONES DASHBOARD 3 ===
def home_player_passmap(ax, df, prog_df, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    if prog_df.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = prog_df['name'].iloc[0]
    acc = df[(df['name']==name)&(df['type']=='Pass')&(df['outcomeType']=='Successful')]
    pro = acc[(acc['prog_pass']>=9.11)&(acc['x']>=35)&(~acc['qualifiers'].str.contains('CornerTaken|Freekick',na=False))]
    car = df[(df['name']==name)&(df['prog_carry']>=9.11)&(df['endX']>=35)]
    key = acc[acc['qualifiers'].str.contains('KeyPass',na=False)]
    pitch.lines(acc.x,acc.y,acc.endX,acc.endY,color=line_color,lw=2,alpha=0.15,comet=True,zorder=2,ax=ax)
    pitch.lines(pro.x,pro.y,pro.endX,pro.endY,color=col,lw=3,alpha=1,comet=True,zorder=3,ax=ax)
    pitch.lines(key.x,key.y,key.endX,key.endY,color=violet,lw=4,alpha=1,comet=True,zorder=4,ax=ax)
    ax.scatter(acc.endX,acc.endY,s=30,color=bg_color,edgecolor='gray',zorder=2)
    ax.scatter(pro.endX,pro.endY,s=40,color=bg_color,edgecolor=col,zorder=3)
    ax.scatter(key.endX,key.endY,s=50,color=bg_color,edgecolor=violet,zorder=4)
    for _,r in car.iterrows():
        if pd.notna(r['x']) and pd.notna(r['endX']):
            ax.add_patch(patches.FancyArrowPatch((r['x'],r['y']),(r['endX'],r['endY']),arrowstyle='->',color=col,zorder=4,mutation_scale=20,alpha=0.9,linewidth=2,linestyle='--'))
    ax.set_title(f"{prog_df['shortName'].iloc[0]} Mapa de pases",color=col,fontsize=25,fontweight='bold',y=1.03)
    ax.text(0,-3,f'Pase Prog: {len(pro)}     Conduccion Prog: {len(car)}',fontsize=13,color=col,ha='left')

def away_player_passmap(ax, df, prog_df, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    ax.invert_xaxis(); ax.invert_yaxis()
    if prog_df.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = prog_df['name'].iloc[0]
    acc = df[(df['name']==name)&(df['type']=='Pass')&(df['outcomeType']=='Successful')]
    pro = acc[(acc['prog_pass']>=9.11)&(acc['x']>=35)&(~acc['qualifiers'].str.contains('CornerTaken|Freekick',na=False))]
    car = df[(df['name']==name)&(df['prog_carry']>=9.11)&(df['endX']>=35)]
    key = acc[acc['qualifiers'].str.contains('KeyPass',na=False)]
    pitch.lines(acc.x,acc.y,acc.endX,acc.endY,color=line_color,lw=2,alpha=0.15,comet=True,zorder=2,ax=ax)
    pitch.lines(pro.x,pro.y,pro.endX,pro.endY,color=col,lw=3,alpha=1,comet=True,zorder=3,ax=ax)
    pitch.lines(key.x,key.y,key.endX,key.endY,color=violet,lw=4,alpha=1,comet=True,zorder=4,ax=ax)
    ax.scatter(acc.endX,acc.endY,s=30,color=bg_color,edgecolor='gray',zorder=2)
    ax.scatter(pro.endX,pro.endY,s=40,color=bg_color,edgecolor=col,zorder=3)
    ax.scatter(key.endX,key.endY,s=50,color=bg_color,edgecolor=violet,zorder=4)
    for _,r in car.iterrows():
        if pd.notna(r['x']) and pd.notna(r['endX']):
            ax.add_patch(patches.FancyArrowPatch((r['x'],r['y']),(r['endX'],r['endY']),arrowstyle='->',color=col,zorder=4,mutation_scale=20,alpha=0.9,linewidth=2,linestyle='--'))
    ax.set_title(f"{prog_df['shortName'].iloc[0]} Mapa de pases",color=col,fontsize=25,fontweight='bold',y=1.03)
    ax.text(105,71,f'Pase Prog: {len(pro)}     Conduccion Prog: {len(car)}',fontsize=13,color=col,ha='left')

def home_passes_received(ax, df, avg_locs, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    fw = avg_locs[avg_locs['position']=='FW']
    if fw.empty: fw = avg_locs.nlargest(1,'pass_avg_x')
    if fw.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = fw['name'].iloc[0]; sn = get_short_name(name)
    filt = df[(df['type']=='Pass')&(df['outcomeType']=='Successful')&(df['name'].shift(-1)==name)]
    key = filt[filt['qualifiers'].str.contains('KeyPass',na=False)]
    pitch.lines(filt.x,filt.y,filt.endX,filt.endY,lw=3,comet=True,color=col,ax=ax,alpha=0.5)
    pitch.lines(key.x,key.y,key.endX,key.endY,lw=4,comet=True,color=violet,ax=ax,alpha=0.75)
    pitch.scatter(filt.endX,filt.endY,s=30,edgecolor=col,linewidth=1,color=bg_color,zorder=2,ax=ax)
    ax.set_title(f"{sn} Pases Recibidos",color=col,fontsize=25,fontweight='bold',y=1.03)
    ax.text(52.5,-3,f'Pases recibidos: {len(filt)} | Pases claves: {len(key)}',color=line_color,fontsize=13,ha='center')

def away_passes_received(ax, df, avg_locs, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    ax.invert_xaxis(); ax.invert_yaxis()
    fw = avg_locs[avg_locs['position']=='FW']
    if fw.empty: fw = avg_locs.nlargest(1,'pass_avg_x')
    if fw.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = fw['name'].iloc[0]; sn = get_short_name(name)
    filt = df[(df['type']=='Pass')&(df['outcomeType']=='Successful')&(df['name'].shift(-1)==name)]
    key = filt[filt['qualifiers'].str.contains('KeyPass',na=False)]
    pitch.lines(filt.x,filt.y,filt.endX,filt.endY,lw=3,comet=True,color=col,ax=ax,alpha=0.5)
    pitch.lines(key.x,key.y,key.endX,key.endY,lw=4,comet=True,color=violet,ax=ax,alpha=0.75)
    pitch.scatter(filt.endX,filt.endY,s=30,edgecolor=col,linewidth=1,color=bg_color,zorder=2,ax=ax)
    ax.set_title(f"{sn} Pases Recibidos",color=col,fontsize=25,fontweight='bold',y=1.03)
    ax.text(52.5,71,f'Pases recibidos: {len(filt)} | Pases claves: {len(key)}',color=line_color,fontsize=13,ha='center')

def home_player_def_acts(ax, df, def_df, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-12,68.5)
    if def_df.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = def_df['name'].iloc[0]
    pdf = df[df['name']==name]
    tk = pdf[pdf['type']=='Tackle']; intc = pdf[(pdf['type']=='Interception')|(pdf['type']=='BlockedPass')]
    br = pdf[pdf['type']=='BallRecovery']; cl = pdf[pdf['type']=='Clearance']
    pitch.scatter(tk.x,tk.y,s=250,c=col,lw=2.5,edgecolor=col,marker='+',ax=ax)
    pitch.scatter(intc.x,intc.y,s=250,c='None',lw=2.5,edgecolor=col,marker='s',hatch='/////',ax=ax)
    pitch.scatter(br.x,br.y,s=250,c='None',lw=2.5,edgecolor=col,marker='o',hatch='/////',ax=ax)
    pitch.scatter(cl.x,cl.y,s=250,c='None',lw=2.5,edgecolor=col,marker='d',hatch='/////',ax=ax)
    ax.text(5,-5,f"Ent:{len(tk)} Int:{len(intc)} Rec:{len(br)} Desp:{len(cl)}",color=col,fontsize=12)
    ax.set_title(f"{def_df['shortName'].iloc[0]} Acciones Defensivas",color=col,fontsize=25,fontweight='bold')

def away_player_def_acts(ax, df, def_df, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,80)
    ax.invert_xaxis(); ax.invert_yaxis()
    if def_df.empty: ax.set_title("Sin datos",color=col,fontsize=25,fontweight='bold'); return
    name = def_df['name'].iloc[0]
    pdf = df[df['name']==name]
    tk = pdf[pdf['type']=='Tackle']; intc = pdf[(pdf['type']=='Interception')|(pdf['type']=='BlockedPass')]
    br = pdf[pdf['type']=='BallRecovery']; cl = pdf[pdf['type']=='Clearance']
    pitch.scatter(tk.x,tk.y,s=250,c=col,lw=2.5,edgecolor=col,marker='+',ax=ax)
    pitch.scatter(intc.x,intc.y,s=250,c='None',lw=2.5,edgecolor=col,marker='s',hatch='/////',ax=ax)
    pitch.scatter(br.x,br.y,s=250,c='None',lw=2.5,edgecolor=col,marker='o',hatch='/////',ax=ax)
    pitch.scatter(cl.x,cl.y,s=250,c='None',lw=2.5,edgecolor=col,marker='d',hatch='/////',ax=ax)
    ax.text(5,73,f"Ent:{len(tk)} Int:{len(intc)} Rec:{len(br)} Desp:{len(cl)}",color=col,fontsize=12,ha='right')
    ax.set_title(f"{def_df['shortName'].iloc[0]} Acciones Defensivas",color=col,fontsize=25,fontweight='bold')

def home_gk_passmap(ax, df, teamName, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    gkdf = df[(df['teamName']==teamName)&(df['position']=='GK')]
    gkp = gkdf[gkdf['type']=='Pass']
    if gkp.empty: ax.set_title("Sin datos PT",color=col,fontsize=25,fontweight='bold'); return
    gkn = get_short_name(gkdf['name'].iloc[0]) if 'name' in gkdf.columns else 'PT'
    for _,r in gkp.iterrows():
        c = col if r['outcomeType']=='Successful' else 'gray'
        pitch.lines(r['x'],r['y'],r['endX'],r['endY'],color=c,lw=3,comet=True,alpha=0.6,zorder=2,ax=ax)
        ax.scatter(r['endX'],r['endY'],s=30,color=bg_color if r['outcomeType']!='Successful' else c,edgecolor=c,zorder=3)
    ax.set_title(f'{gkn} Mapa de Pases Portero',color=col,fontsize=25,fontweight='bold')
    ax.text(52.5,-3,f'Pases: {len(gkp)} | Ok: {len(gkp[gkp["outcomeType"]=="Successful"])}',color=line_color,fontsize=13,ha='center')

def away_gk_passmap(ax, df, teamName, col, bg_color, line_color):
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax); ax.set_xlim(-0.5,105.5); ax.set_ylim(-0.5,68.5)
    ax.invert_xaxis(); ax.invert_yaxis()
    gkdf = df[(df['teamName']==teamName)&(df['position']=='GK')]
    gkp = gkdf[gkdf['type']=='Pass']
    if gkp.empty: ax.set_title("Sin datos PT",color=col,fontsize=25,fontweight='bold'); return
    gkn = get_short_name(gkdf['name'].iloc[0]) if 'name' in gkdf.columns else 'PT'
    for _,r in gkp.iterrows():
        c = col if r['outcomeType']=='Successful' else 'gray'
        pitch.lines(r['x'],r['y'],r['endX'],r['endY'],color=c,lw=3,comet=True,alpha=0.6,zorder=2,ax=ax)
        ax.scatter(r['endX'],r['endY'],s=30,color=bg_color if r['outcomeType']!='Successful' else c,edgecolor=c,zorder=3)
    ax.set_title(f'{gkn} Mapa de Pases Portero',color=col,fontsize=25,fontweight='bold')
    ax.text(52.5,71,f'Pases: {len(gkp)} | Ok: {len(gkp[gkp["outcomeType"]=="Successful"])}',color=line_color,fontsize=13,ha='center')

def passer_bar(ax, hp, ap, hcol, acol, bg_color, line_color):
    pf = pd.concat([hp,ap]).sort_values('total',ascending=False).head(10).sort_values('total',ascending=True)
    if pf.empty: ax.set_facecolor(bg_color); ax.set_title("Top 10 progresores",color=line_color,fontsize=25,fontweight='bold'); return
    n,pp,pc = pf['shortName'].tolist(),pf['Progressive Passes'].tolist(),pf['Progressive Carries'].tolist()
    ax.barh(n,pp,label='Pases Prog',color=hcol); ax.barh(n,pc,label='Cond Prog',color=acol,left=pp)
    for i,(p,c) in enumerate(zip(pp,pc)):
        if p>0: ax.text(p/2,i,str(int(p)),ha='center',va='center',color=line_color,fontsize=11,fontweight='bold')
        if c>0: ax.text(p+c/2,i,str(int(c)),ha='center',va='center',color=line_color,fontsize=11,fontweight='bold')
    ax.set_facecolor(bg_color); ax.tick_params(axis='x',colors=line_color,labelsize=12); ax.tick_params(axis='y',colors=line_color,labelsize=12)
    for s in ax.spines.values(): s.set_edgecolor(bg_color)
    ax.set_title("Top 10 progresores de balón",color=line_color,fontsize=25,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10,facecolor=bg_color,labelcolor=line_color)

def defender_bar(ax, hd, ad, hcol, acol, bg_color, line_color):
    df2 = pd.concat([hd,ad]).sort_values('total',ascending=False).head(10).sort_values('total',ascending=True)
    if df2.empty: ax.set_facecolor(bg_color); ax.set_title("Top10 Defensores",color=line_color,fontsize=25,fontweight='bold'); return
    n,tk,i,c = df2['shortName'].tolist(),df2['Tackles'].tolist(),df2['Interceptions'].tolist(),df2['Clearance'].tolist()
    l1 = [t+x for t,x in zip(tk,i)]
    ax.barh(n,tk,label='Entradas',color=hcol); ax.barh(n,i,label='Interc',color=violet,left=tk); ax.barh(n,c,label='Desp',color=acol,left=l1)
    ax.set_facecolor(bg_color); ax.tick_params(axis='x',colors=line_color,labelsize=12); ax.tick_params(axis='y',colors=line_color,labelsize=12)
    for s in ax.spines.values(): s.set_edgecolor(bg_color)
    ax.set_title("Top10 Defensores",color=line_color,fontsize=25,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10,facecolor=bg_color,labelcolor=line_color)

def shot_seq_bar(ax, hs, aws, hcol, acol, bg_color, line_color):
    df2 = pd.concat([hs,aws]).sort_values('total',ascending=False).head(10).sort_values('total',ascending=True)
    if df2.empty: ax.set_facecolor(bg_color); ax.set_title("Secuencias de tiros",color=line_color,fontsize=25,fontweight='bold'); return
    n,sh,sa,bu = df2['shortName'].tolist(),df2['Shots'].tolist(),df2['Shot Assist'].tolist(),df2['Buildup to shot'].tolist()
    l1 = [s+a for s,a in zip(sh,sa)]
    ax.barh(n,sh,label='Tiro',color=hcol); ax.barh(n,sa,label='Asist',color=violet,left=sh); ax.barh(n,bu,label='Constr',color=acol,left=l1)
    ax.set_facecolor(bg_color); ax.tick_params(axis='x',colors=line_color,labelsize=12); ax.tick_params(axis='y',colors=line_color,labelsize=12)
    for s in ax.spines.values(): s.set_edgecolor(bg_color)
    ax.set_title("Participación en secuencias de tiros",color=line_color,fontsize=25,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10,facecolor=bg_color,labelcolor=line_color)

# === GENERACIÓN DE DASHBOARDS ===
def generate_dashboard_1(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst, analyst_col, hg, ag):
    fig,axs = plt.subplots(4,3,figsize=(35,35),facecolor=bg_color)
    passes_df = get_passes_df(df)
    hpb,hal = get_passes_between_df(hteamName,passes_df,df,players_df)
    apb,aal = get_passes_between_df(ateamName,passes_df,df,players_df)
    da_df = get_defensive_action_df(df)
    hdal = get_da_count_df(hteamName,da_df,players_df); hdal = hdal[hdal['position']!='GK']
    adal = get_da_count_df(ateamName,da_df,players_df); adal = adal[adal['position']!='GK']
    pass_network_visualization(axs[0,0],hpb,hal,hcol,hteamName,hteamName,bg_color,line_color,passes_df)
    plot_shotmap(axs[0,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    pass_network_visualization(axs[0,2],apb,aal,acol,ateamName,hteamName,bg_color,line_color,passes_df)
    defensive_block(axs[1,0],hdal,da_df,hteamName,hcol,hteamName,bg_color,line_color)
    plot_goalPost(axs[1,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    defensive_block(axs[1,2],adal,da_df,ateamName,acol,hteamName,bg_color,line_color)
    draw_progressive_pass_map(axs[2,0],df,hteamName,hcol,hteamName,bg_color,line_color)
    axs[2,1].set_facecolor(bg_color); axs[2,1].axis('off')
    axs[2,1].text(0.5,0.5,"Momento del partido por xT\n(Requiere datos FotMob)",color=line_color,fontsize=20,ha='center',va='center',transform=axs[2,1].transAxes)
    draw_progressive_pass_map(axs[2,2],df,ateamName,acol,hteamName,bg_color,line_color)
    draw_progressive_carry_map(axs[3,0],df,hteamName,hcol,hteamName,bg_color,line_color)
    plotting_match_stats(axs[3,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    draw_progressive_carry_map(axs[3,2],df,ateamName,acol,hteamName,bg_color,line_color)
    fig.text(0.5,0.98,f"{hteamName} {hg} - {ag} {ateamName}",color=line_color,fontsize=60,fontweight='bold',ha='center',va='top')
    fig.text(0.5,0.95,titulo,color=line_color,fontsize=28,ha='center',va='top')
    fig.text(0.5,0.93,f"{subtitulo} | Analista {analyst}",color=analyst_col,fontsize=22,ha='center',va='top')
    fig.text(0.125,0.08,'Direccion Ataque --->',color=hcol,fontsize=22,ha='left')
    fig.text(0.875,0.08,'<--- Direccion Ataque',color=acol,fontsize=22,ha='right')
    plt.tight_layout(rect=[0,0.1,1,0.92])
    return fig

def generate_dashboard_2(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst, analyst_col, hg, ag):
    fig,axs = plt.subplots(4,3,figsize=(35,35),facecolor=bg_color)
    Final_third_entry(axs[0,0],df,hteamName,hcol,hteamName,bg_color,line_color)
    box_entry(axs[0,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    Final_third_entry(axs[0,2],df,ateamName,acol,hteamName,bg_color,line_color)
    zone14hs(axs[1,0],df,hteamName,hcol,hteamName,bg_color,line_color)
    Crosses(axs[1,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    zone14hs(axs[1,2],df,ateamName,acol,hteamName,bg_color,line_color)
    Pass_end_zone(axs[2,0],df,hteamName,hcol,hteamName,bg_color,line_color)
    HighTO(axs[2,1],df,hteamName,ateamName,hcol,acol,bg_color,line_color)
    Pass_end_zone(axs[2,2],df,ateamName,acol,hteamName,bg_color,line_color)
    for i,(t,c) in enumerate([(hteamName,hcol),(ateamName,acol)]):
        ai = 0 if i==0 else 2
        axs[3,ai].set_facecolor(bg_color); axs[3,ai].axis('off')
        axs[3,ai].text(0.5,0.5,f"{t}\nZona Oportunidades creadas\n(Requiere datos adicionales)",color=c,fontsize=16,ha='center',va='center',transform=axs[3,ai].transAxes)
    axs[3,1].set_facecolor(bg_color); axs[3,1].axis('off')
    axs[3,1].text(0.5,0.5,"Zona de dominio equipo\n(Requiere cálculos territoriales)",color=line_color,fontsize=16,ha='center',va='center',transform=axs[3,1].transAxes)
    fig.text(0.5,0.98,f"{hteamName} {hg} - {ag} {ateamName}",color=line_color,fontsize=60,fontweight='bold',ha='center',va='top')
    fig.text(0.5,0.95,titulo.replace("Informe-1","Informe-2"),color=line_color,fontsize=28,ha='center',va='top')
    fig.text(0.5,0.93,f"{subtitulo} | Analista {analyst}",color=analyst_col,fontsize=22,ha='center',va='top')
    fig.text(0.125,0.08,'Direccion Ataque --->',color=hcol,fontsize=22,ha='left')
    fig.text(0.875,0.08,'<--- Direccion Ataque',color=acol,fontsize=22,ha='right')
    plt.tight_layout(rect=[0,0.1,1,0.92])
    return fig

def generate_dashboard_3(df, players_df, hteamName, ateamName, hcol, acol, bg_color, line_color, titulo, subtitulo, analyst, analyst_col, hg, ag):
    fig,axs = plt.subplots(4,3,figsize=(35,35),facecolor=bg_color)
    hp = create_progressor_df(df,hteamName); ap = create_progressor_df(df,ateamName)
    hd = create_defender_df(df,hteamName); ad = create_defender_df(df,ateamName)
    hs = create_shot_sequence_df(df,hteamName); aws = create_shot_sequence_df(df,ateamName)
    passes_df = get_passes_df(df)
    _,hal = get_passes_between_df(hteamName,passes_df,df,players_df)
    _,aal = get_passes_between_df(ateamName,passes_df,df,players_df)
    home_player_passmap(axs[0,0],df,hp,hcol,bg_color,line_color)
    passer_bar(axs[0,1],hp,ap,hcol,acol,bg_color,line_color)
    away_player_passmap(axs[0,2],df,ap,acol,bg_color,line_color)
    home_passes_received(axs[1,0],df,hal,hcol,bg_color,line_color)
    shot_seq_bar(axs[1,1],hs,aws,hcol,acol,bg_color,line_color)
    away_passes_received(axs[1,2],df,aal,acol,bg_color,line_color)
    home_player_def_acts(axs[2,0],df,hd,hcol,bg_color,line_color)
    defender_bar(axs[2,1],hd,ad,hcol,acol,bg_color,line_color)
    away_player_def_acts(axs[2,2],df,ad,acol,bg_color,line_color)
    home_gk_passmap(axs[3,0],df,hteamName,hcol,bg_color,line_color)
    axs[3,1].set_facecolor(bg_color); axs[3,1].axis('off')
    axs[3,1].text(0.5,0.5,"Top 10 jugadores más peligrosos\n(Requiere cálculo xT)",color=line_color,fontsize=16,ha='center',va='center',transform=axs[3,1].transAxes)
    away_gk_passmap(axs[3,2],df,ateamName,acol,bg_color,line_color)
    fig.text(0.5,0.98,f"{hteamName} {hg} - {ag} {ateamName}",color=line_color,fontsize=60,fontweight='bold',ha='center',va='top')
    fig.text(0.5,0.95,titulo.replace("Post Partido Informe-1","Top Jugadores del Partido"),color=line_color,fontsize=28,ha='center',va='top')
    fig.text(0.5,0.93,f"{subtitulo} | Analista {analyst}",color=analyst_col,fontsize=22,ha='center',va='top')
    fig.text(0.125,0.08,'Direccion Ataque --->',color=hcol,fontsize=22,ha='left')
    fig.text(0.875,0.08,'<--- Direccion Ataque',color=acol,fontsize=22,ha='right')
    plt.tight_layout(rect=[0,0.1,1,0.92])
    return fig

# === APLICACIÓN PRINCIPAL ===
def main():
    st.title("⚽ Zona del Analista - Dashboard Generator")
    with st.sidebar:
        st.header("📁 Cargar Datos")
        uploaded_file = st.file_uploader("HTML de WhoScored",type=['html','htm'])
        st.divider(); st.header("📝 Información")
        titulo = st.text_input("Título","Champions 25/26 Jornada 5| Post Partido Informe-1")
        subtitulo = st.text_input("Subtítulo","Miercoles 26/11/25")
        analyst = st.text_input("Analista","John Triguero")
        st.divider(); st.header("🎨 Colores")
        bg_color = st.color_picker("Fondo",'#363d4d')
        line_color = st.color_picker("Líneas",'#ffffff')
        hcol = st.color_picker("Eq Local",'#ff4b44')
        acol = st.color_picker("Eq Visitante",'#00FFD5')
        analyst_col = st.color_picker("Analista",'#ffffff')
    if uploaded_file:
        try:
            html = uploaded_file.read().decode('utf-8')
            with st.spinner('Procesando...'):
                jd = extract_json_from_html(html)
                data = json.loads(jd)
                events,players_df,teams_dict = extract_data_from_dict(data)
                df = pd.DataFrame(events)
                df = process_dataframe(df,teams_dict)
                df = insert_ball_carries(df)
                df = df.merge(players_df[['playerId','position','name']],on='playerId',how='left',suffixes=('','_p'))
                if 'name_p' in df.columns: df['name']=df['name'].fillna(df['name_p']); df.drop(columns=['name_p'],inplace=True)
                if 'position_p' in df.columns: df['position']=df['position'].fillna(df['position_p']); df.drop(columns=['position_p'],inplace=True)
                tids = list(teams_dict.keys())
                hteam,ateam = teams_dict[tids[0]],teams_dict[tids[1]]
                hdf,adf = df[df['teamName']==hteam],df[df['teamName']==ateam]
                hg = len(hdf[(hdf['type']=='Goal')&(~hdf['qualifiers'].str.contains('OwnGoal',na=False))])+len(adf[(adf['type']=='Goal')&(adf['qualifiers'].str.contains('OwnGoal',na=False))])
                ag = len(adf[(adf['type']=='Goal')&(~adf['qualifiers'].str.contains('OwnGoal',na=False))])+len(hdf[(hdf['type']=='Goal')&(hdf['qualifiers'].str.contains('OwnGoal',na=False))])
            st.success(f"✅ {hteam} {hg} - {ag} {ateam}")
            c1,c2,c3 = st.columns(3)
            with c1:
                if st.button("📊 Dashboard 1\n(Informe General)",use_container_width=True):
                    with st.spinner('Generando...'): 
                        fig = generate_dashboard_1(df,players_df,hteam,ateam,hcol,acol,bg_color,line_color,titulo,subtitulo,analyst,analyst_col,hg,ag)
                        buf = BytesIO(); fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor=bg_color); buf.seek(0)
                        st.download_button("📥 Descargar",buf,f"{hteam}_vs_{ateam}_D1.png","image/png")
                        st.pyplot(fig); plt.close()
            with c2:
                if st.button("📊 Dashboard 2\n(Zonas)",use_container_width=True):
                    with st.spinner('Generando...'): 
                        fig = generate_dashboard_2(df,players_df,hteam,ateam,hcol,acol,bg_color,line_color,titulo,subtitulo,analyst,analyst_col,hg,ag)
                        buf = BytesIO(); fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor=bg_color); buf.seek(0)
                        st.download_button("📥 Descargar",buf,f"{hteam}_vs_{ateam}_D2.png","image/png")
                        st.pyplot(fig); plt.close()
            with c3:
                if st.button("📊 Dashboard 3\n(Top Jugadores)",use_container_width=True):
                    with st.spinner('Generando...'): 
                        fig = generate_dashboard_3(df,players_df,hteam,ateam,hcol,acol,bg_color,line_color,titulo,subtitulo,analyst,analyst_col,hg,ag)
                        buf = BytesIO(); fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor=bg_color); buf.seek(0)
                        st.download_button("📥 Descargar",buf,f"{hteam}_vs_{ateam}_D3.png","image/png")
                        st.pyplot(fig); plt.close()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback; st.code(traceback.format_exc())
    else: st.info("👆 Sube un HTML de WhoScored para comenzar")

if __name__ == "__main__": main()
