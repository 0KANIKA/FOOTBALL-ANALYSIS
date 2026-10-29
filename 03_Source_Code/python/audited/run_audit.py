from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'outputs'; OUT.mkdir(exist_ok=True)
SEED=42
TARGETS=['post_GA90','post_goals_per90','post_assists_per90','post_minutes','post_appearances']
# Stable cohorts, feature groups, and a modest validation-only search.
CORE_NUM=['age_at_transfer','height_in_cm','appearances_pre','minutes_pre','goals_pre','assists_pre','goals_per90_pre','assists_per90_pre','yellow_cards_pre','red_cards_pre','transfer_market_value','market_value_pre','transfer_fee']
CORE_CAT=['position','sub_position','foot']
TEAM=['rich_team_matches_last10','rich_team_goals_for_avg10','rich_team_goals_against_avg10','rich_team_goal_difference_avg10','rich_team_win_rate10','rich_team_points_per_game10']
TACTIC_NUM=['dominant_formation_frequency10','formation_diversity10','manager_matches_last10','unique_managers_last10']
TACTIC_CAT=['dominant_formation10','dominant_manager10']
SQUAD=['same_position_competitor_count','same_position_competitor_minutes','same_position_avg_market_value','same_position_top_competitor_minutes','same_sub_position_competitor_count','same_sub_position_competitor_minutes','same_sub_position_top_minutes','position_minutes_share']
COACH=['coach_known_pretransfer','coach_position_group_minutes_share','coach_position_group_start_rate','coach_exact_position_minutes_share','coach_exact_position_start_rate','coach_position_competitor_count','coach_position_competitor_minutes','coach_position_rotation_hhi','coach_formation_stability','coach_observed_tenure_days']
GROUPS={'core':(CORE_NUM,CORE_CAT),'mode_A':(CORE_NUM+TEAM,CORE_CAT),'full_enhanced':(CORE_NUM+TEAM+TACTIC_NUM+SQUAD+COACH,CORE_CAT+TACTIC_CAT),'without_tactical_manager':(CORE_NUM+TEAM+SQUAD+COACH,CORE_CAT),'without_squad_competition':(CORE_NUM+TEAM+TACTIC_NUM+COACH,CORE_CAT+TACTIC_CAT),'without_coach':(CORE_NUM+TEAM+TACTIC_NUM+SQUAD,CORE_CAT+TACTIC_CAT)}

def read(name): return pd.read_csv(ROOT/name, low_memory=False)
base=read('post_transfer_performance.csv')
rich=read('rich_team_context_dataset.csv')
coach=read('coach_tactical_context_v21.csv')
# Context assets are indexed by a generated 1-based transfer_row_id. Validate key uniqueness before joining.
for label, frame in [('target',base),('rich',rich),('coach',coach)]:
 if 'transfer_row_id' not in frame: raise ValueError(f'{label}: missing transfer_row_id')
 dup=int(frame.transfer_row_id.duplicated().sum())
 if dup: raise ValueError(f'{label}: duplicate transfer_row_id={dup}; refusing many-to-many merge')
df=base.merge(rich[['transfer_row_id']+[c for c in TEAM+TACTIC_NUM+TACTIC_CAT+SQUAD if c in rich]],on='transfer_row_id',how='left',validate='one_to_one')
df=df.merge(coach,on='transfer_row_id',how='left',validate='one_to_one')
df['transfer_date']=pd.to_datetime(df.transfer_date,errors='coerce')
# Retain only complete forward windows; eligible_450min makes post-rate targets well-defined.
initial=len(df)
df=df[(df.complete_365_day_window==1)&(df.eligible_450min==1)&df.transfer_date.notna()].copy()
cohort_after_elig=len(df)
# Exact dedup only; distinct transfers by same player are valid independent events.
identity=['player_id','transfer_date','from_club_id','to_club_id']
exact_dups=int(df.duplicated(identity,keep='first').sum())
df=df.drop_duplicates(identity,keep='first').sort_values(['transfer_date','transfer_row_id'],kind='mergesort').reset_index(drop=True)
if 'post_goals_per90' not in df: df['post_goals_per90']=np.where(df.post_minutes>0,df.post_goals/df.post_minutes*90,np.nan)
if 'post_assists_per90' not in df: df['post_assists_per90']=np.where(df.post_minutes>0,df.post_assists/df.post_minutes*90,np.nan)
# Use one common cohort with finite primary target, then identical index boundaries for all models.
df=df[np.isfinite(pd.to_numeric(df.post_GA90,errors='coerce'))].reset_index(drop=True)
n=len(df)
# Split at whole-date boundaries nearest 60% and 80%, preventing same-day transfers crossing partitions.
daily=df.groupby('transfer_date',sort=True).size().cumsum()
i=int(daily.index[np.argmin(np.abs(daily.to_numpy()-.60*n))] is None) if False else int(np.searchsorted(daily.to_numpy(),.60*n,side='left')+1)
j=int(np.searchsorted(daily.to_numpy(),.80*n,side='left')+1)
i=min(max(i,1),len(daily)-2); j=min(max(j,i+1),len(daily)-1)
date_train_end=daily.index[i-1]; date_val_end=daily.index[j-1]
parts={'train':df[df.transfer_date<=date_train_end].copy(),'validation':df[(df.transfer_date>date_train_end)&(df.transfer_date<=date_val_end)].copy(),'test':df[df.transfer_date>date_val_end].copy()}
i=len(parts['train']); j=i+len(parts['validation'])
# Reject any predictor absent from its source dataset, and prohibit post-outcome columns.
for name,(nums,cats) in GROUPS.items():
 missing=[c for c in nums+cats if c not in df]
 if missing: raise ValueError(f'{name} missing columns: {missing}')
 forbidden=set(TARGETS+['post_goals','post_assists','post_minutes','post_appearances','window_end','complete_365_day_window','eligible_450min','transfer_row_id'])
 if forbidden.intersection(nums+cats): raise ValueError(f'Outcome/administrative leakage in {name}')

# Train-only imputation/encoding. Tune mtry-equivalent max_features and leaf size by validation MAE only.
search=[('sqrt',5),(.5,10)]
metrics=[]; preds=[]; tuning=[]
for target in TARGETS:
 yall=pd.to_numeric(df[target],errors='coerce')
 # Missing secondary rate targets can reflect zero minutes; use valid target rows and record per-target counts.
 for splitname,part in parts.items():
  pass
 valid=np.isfinite(yall.to_numpy())
 split_mask={k:valid[v.index] for k,v in parts.items()}
 # Keep exact same underlying train/validation/test records across every model for this target.
 tr=parts['train'].loc[split_mask['train']]; va=parts['validation'].loc[split_mask['validation']]; te=parts['test'].loc[split_mask['test']]
 for model_name,(nums,cats) in GROUPS.items():
  best=None
  for maxf,leaf in search:
   prep=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('ohe',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=True))]),cats)],remainder='drop')
   model=Pipeline([('prep',prep),('rf',RandomForestRegressor(n_estimators=30,max_features=maxf,min_samples_leaf=leaf,random_state=SEED,n_jobs=4))])
   model.fit(tr[nums+cats],yall.loc[tr.index])
   pv=model.predict(va[nums+cats]); score=mean_absolute_error(yall.loc[va.index],pv)
   tuning.append({'target':target,'model':model_name,'max_features':str(maxf),'min_samples_leaf':leaf,'validation_MAE':score,'n_train':len(tr),'n_validation':len(va)})
   if best is None or score<best[0]: best=(score,maxf,leaf)
  _,maxf,leaf=best
  prep=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('ohe',OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=True))]),cats)],remainder='drop')
  model=Pipeline([('prep',prep),('rf',RandomForestRegressor(n_estimators=60,max_features=maxf,min_samples_leaf=leaf,random_state=SEED,n_jobs=4))])
  # Refit on train+validation after hyperparameter choice; test remains untouched until final prediction.
  fit=pd.concat([tr,va]); model.fit(fit[nums+cats],yall.loc[fit.index])
  for splitname,part in [('validation',va),('test',te)]:
   actual=yall.loc[part.index].to_numpy(); pred=model.predict(part[nums+cats])
   metrics.append({'target':target,'model':model_name,'split':splitname,'n':len(part),'MAE':mean_absolute_error(actual,pred),'RMSE':mean_squared_error(actual,pred)**.5,'R2':r2_score(actual,pred),'max_features':str(maxf),'min_samples_leaf':leaf})
   if target=='post_GA90':
    preds.append(pd.DataFrame({'transfer_row_id':part.transfer_row_id,'transfer_date':part.transfer_date,'player_id':part.player_id,'actual':actual,'prediction':pred,'model':model_name,'split':splitname}))

pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
pd.DataFrame(tuning).to_csv(OUT/'validation_tuning.csv',index=False)
pd.concat(preds,ignore_index=True).to_csv(OUT/'post_GA90_predictions.csv',index=False)
# Audit cohort + temporal split metadata
rows=[]
for k,p in parts.items(): rows.append({'split':k,'n_primary_cohort':len(p),'min_date':p.transfer_date.min(),'max_date':p.transfer_date.max(),'missing_post_minutes':int(p.post_minutes.isna().sum()),'zero_post_minutes':int((p.post_minutes==0).sum()),'missing_post_GA90':int(p.post_GA90.isna().sum())})
pd.DataFrame(rows).to_csv(OUT/'split_audit.csv',index=False)
coverage=pd.DataFrame([{'feature':c,'missing_n':int(df[c].isna().sum()),'missing_pct':float(df[c].isna().mean()*100),'unique_n':int(df[c].nunique(dropna=True))} for c in sum(([n for n in v[0]]+v[1] for v in GROUPS.values()),[]) if c in df]).drop_duplicates('feature')
coverage.to_csv(OUT/'feature_coverage.csv',index=False)
(OUT/'run_metadata.json').write_text(json.dumps({'seed':SEED,'chronological_split':[.6,.2,.2],'split_boundary':'whole transfer dates; nearest cumulative 60/80 percent','initial_joined_rows':initial,'eligible_full_window_rows':cohort_after_elig,'primary_common_cohort_rows':len(df),'exact_duplicate_transfer_rows_removed':exact_dups,'n_train':i,'n_validation':j-i,'n_test':n-j,'tuning_rule':'validation MAE, no test feedback','fit_rule':'refit chosen setting on train+validation before final test'},indent=2))
print(pd.DataFrame(metrics).query("target == 'post_GA90'").to_string(index=False))
