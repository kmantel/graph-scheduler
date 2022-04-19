Search.setIndex({docnames:["Condition","Scheduler","Time","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["Condition.rst","Scheduler.rst","Time.rst","index.rst"],objects:{"graph_scheduler.condition":[[0,1,1,"","AfterCall"],[0,1,1,"","AfterConsiderationSetExecution"],[0,1,1,"","AfterEnvironmentSequence"],[0,1,1,"","AfterEnvironmentStateUpdate"],[0,1,1,"","AfterNCalls"],[0,1,1,"","AfterNCallsCombined"],[0,1,1,"","AfterNConsiderationSetExecutions"],[0,1,1,"","AfterNEnvironmentSequences"],[0,1,1,"","AfterNEnvironmentStateUpdates"],[0,1,1,"","AfterNPasses"],[0,1,1,"","AfterPass"],[0,1,1,"","All"],[0,1,1,"","AllHaveRun"],[0,1,1,"","Always"],[0,2,1,"","And"],[0,1,1,"","Any"],[0,1,1,"","AtConsiderationSetExecution"],[0,1,1,"","AtEnvironmentSequence"],[0,1,1,"","AtEnvironmentSequenceNStart"],[0,1,1,"","AtEnvironmentSequenceStart"],[0,1,1,"","AtEnvironmentStateUpdate"],[0,1,1,"","AtEnvironmentStateUpdateNStart"],[0,1,1,"","AtEnvironmentStateUpdateStart"],[0,1,1,"","AtNCalls"],[0,1,1,"","AtPass"],[0,1,1,"","BeforeConsiderationSetExecution"],[0,1,1,"","BeforeEnvironmentStateUpdate"],[0,1,1,"","BeforeNCalls"],[0,1,1,"","BeforePass"],[0,1,1,"","Condition"],[0,1,1,"","ConditionSet"],[0,1,1,"","EveryNCalls"],[0,1,1,"","EveryNPasses"],[0,1,1,"","JustRan"],[0,1,1,"","NWhen"],[0,1,1,"","Never"],[0,1,1,"","Not"],[0,2,1,"","Or"],[0,1,1,"","Threshold"],[0,1,1,"","TimeInterval"],[0,1,1,"","TimeTermination"],[0,1,1,"","WhenFinished"],[0,1,1,"","WhenFinishedAll"],[0,1,1,"","WhenFinishedAny"],[0,2,1,"","While"],[0,1,1,"","WhileNot"]],"graph_scheduler.condition.Condition":[[0,3,1,"","absolute_fixed_points"],[0,3,1,"","absolute_intervals"],[0,4,1,"","is_satisfied"],[0,2,1,"","owner"]],"graph_scheduler.condition.ConditionSet":[[0,4,1,"","add_condition"],[0,4,1,"","add_condition_set"],[0,2,1,"","conditions"]],"graph_scheduler.condition.Threshold":[[0,2,1,"","atol"],[0,2,1,"","comparator"],[0,2,1,"","custom_parameter_getter"],[0,2,1,"","custom_parameter_validator"],[0,2,1,"","dependency"],[0,2,1,"","indices"],[0,2,1,"","parameter"],[0,2,1,"","rtol"],[0,2,1,"","threshold"]],"graph_scheduler.condition.TimeInterval":[[0,3,1,"","absolute_fixed_points"],[0,3,1,"","absolute_intervals"],[0,2,1,"","end"],[0,2,1,"","end_inclusive"],[0,2,1,"","repeat"],[0,2,1,"","start"],[0,2,1,"","start_inclusive"],[0,2,1,"","unit"]],"graph_scheduler.condition.TimeTermination":[[0,3,1,"","absolute_fixed_points"],[0,2,1,"","start_inclusive"],[0,2,1,"","t"],[0,2,1,"","unit"]],"graph_scheduler.scheduler":[[1,1,1,"","Scheduler"],[1,1,1,"","SchedulingMode"]],"graph_scheduler.scheduler.Scheduler":[[1,4,1,"","_get_absolute_consideration_set_execution_unit"],[1,4,1,"","_init_counts"],[1,4,1,"","add_condition"],[1,4,1,"","add_condition_set"],[1,2,1,"","base_execution_id"],[1,2,1,"","conditions"],[1,2,1,"","consideration_queue"],[1,2,1,"","consideration_queue_indices"],[1,2,1,"","default_absolute_time_unit"],[1,2,1,"","default_execution_id"],[1,4,1,"","end_environment_sequence"],[1,2,1,"","execution_id"],[1,2,1,"","execution_list"],[1,2,1,"","mode"],[1,4,1,"","run"],[1,2,1,"","termination_conds"]],"graph_scheduler.scheduler.SchedulingMode":[[1,2,1,"","EXACT_TIME"],[1,2,1,"","STANDARD"]],"graph_scheduler.time":[[2,1,1,"","Clock"],[2,1,1,"","SimpleTime"],[2,1,1,"","Time"],[2,1,1,"","TimeHistoryTree"],[2,1,1,"","TimeScale"],[2,5,1,"","remove_time_scale_alias"],[2,5,1,"","set_time_scale_alias"]],"graph_scheduler.time.Clock":[[2,4,1,"","_increment_time"],[2,4,1,"","get_time_by_time_scale"],[2,4,1,"","get_total_times_relative"],[2,2,1,"","history"],[2,3,1,"","previous_time"],[2,2,1,"","simple_time"],[2,3,1,"","time"]],"graph_scheduler.time.Time":[[2,4,1,"","_get_by_time_scale"],[2,4,1,"","_increment_by_time_scale"],[2,4,1,"","_reset_by_time_scale"],[2,4,1,"","_set_by_time_scale"],[2,2,1,"","absolute"],[2,2,1,"","absolute_enabled"],[2,2,1,"","absolute_interval"],[2,2,1,"","absolute_time_unit_scale"],[2,2,1,"","consideration_set_execution"],[2,2,1,"","environment_sequence"],[2,2,1,"","environment_state_update"],[2,2,1,"","life"],[2,2,1,"","pass_"]],"graph_scheduler.time.TimeHistoryTree":[[2,2,1,"","child_time_scale"],[2,2,1,"","children"],[2,2,1,"","current_time"],[2,4,1,"","get_total_times_relative"],[2,4,1,"","increment_time"],[2,2,1,"","index"],[2,2,1,"","max_depth"],[2,2,1,"","parent"],[2,2,1,"","previous_time"],[2,2,1,"","time_scale"],[2,2,1,"","total_times"]],"graph_scheduler.time.TimeScale":[[2,2,1,"","CONSIDERATION_SET_EXECUTION"],[2,2,1,"","ENVIRONMENT_SEQUENCE"],[2,2,1,"","ENVIRONMENT_STATE_UPDATE"],[2,2,1,"","LIFE"],[2,2,1,"","PASS"],[2,4,1,"","get_child"],[2,4,1,"","get_parent"]],graph_scheduler:[[0,0,0,"-","condition"],[1,0,0,"-","scheduler"],[2,0,0,"-","time"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","property","Python property"],"4":["py","method","Python method"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:property","4":"py:method","5":"py:function"},terms:{"0":[0,1,2],"01":0,"0m":2,"1":[0,1,2],"10":1,"10m":1,"1m":[1,2],"1st":2,"2":[0,1,3],"3":[0,1],"4":[0,1],"5":[0,2],"5th":2,"8":1,"abstract":3,"case":1,"class":[2,3],"default":[0,1,2],"do":[0,1],"enum":2,"final":0,"float":1,"function":0,"import":[1,3],"int":[0,2],"new":[0,1],"return":[0,1,2],"static":0,"throw":0,"true":[0,1,2],"while":[0,1],A:[0,1,2,3],And:0,As:1,At:[0,1],By:1,For:[0,1,3],If:[0,1,2],In:[0,1],It:0,Not:0,On:1,Or:0,The:[0,1,2,3],There:[0,1],These:1,To:0,_get_absolute_consideration_set_execution_unit:1,_get_by_time_scal:2,_increment_by_time_scal:2,_increment_tim:2,_init_count:1,_reset_by_time_scal:2,_set_by_time_scal:2,ab:0,abov:1,absolut:[0,2,3],absolute_en:2,absolute_fixed_point:0,absolute_interv:[0,2],absolute_time_unit_scal:2,access:[0,2],accur:[0,1],achiev:0,action:[1,2],actual:1,acycl:[1,3],ad:[0,1],add:[0,1],add_condit:[0,1,3],add_condition_set:[0,1],addit:0,addition:1,after:[0,1],aftercal:0,afterconsiderationsetexecut:0,afterenvironmentsequ:0,afterenvironmentstateupd:0,afterncal:[0,1],afterncallscombin:0,afternconsiderationsetexecut:0,afternenvironmentsequ:0,afternenvironmentstateupd:0,afternpass:0,afterpass:0,algorithm:3,alia:[0,2],alias_time_valu:2,all:[0,1,2],allhaverun:[0,1],allow:[0,1,2],alreadi:[0,1],also:[0,1,3],altern:1,alwai:[0,1],among:[0,1],amount:[0,2],an:[0,1,2],ani:[0,1],anoth:[0,1],appear:1,append:0,appli:1,ar:[0,1,2,3],arbitrari:[0,1],arbitrarili:0,arg:0,argument:[0,1],assign:[0,1],associ:[0,1,2],atconsiderationsetexecut:0,atenvironmentsequ:0,atenvironmentsequencenstart:0,atenvironmentsequencestart:0,atenvironmentstateupd:0,atenvironmentstateupdatenstart:0,atenvironmentstateupdatestart:0,atncal:0,atol:0,atpass:[0,1],attribut:[0,2],automat:0,avail:3,b:[0,1,3],back:1,base:[0,1,3],base_execution_id:1,base_index:2,base_indic:2,base_time_scal:2,basi:0,basic:1,batch:2,becaus:1,becom:[0,1],been:[0,1],befor:[0,1],beforeconsiderationsetexecut:0,beforeenvironmentstateupd:0,beforencal:0,beforepass:0,begin:0,behav:[0,1],behavior:[0,1],belong:0,below:[0,1],between:[0,1,2],bool:2,both:1,branch:3,c:[0,1,3],call:[0,1,2],callabl:0,can:[0,1],cannot:2,capabl:1,caus:[0,1],certain:[1,2],chanc:1,chang:0,check:0,child_time_scal:2,children:[1,2],classmethod:2,clock:2,coars:2,coarser:2,collect:[0,1],combin:[0,1],come:1,compar:0,comparison:0,compat:1,complet:1,complex:1,composit:[0,1],composite_condit:0,compris:1,comput:1,condens:0,condit:[2,3],conditionset:[0,1],conjunct:[0,2],consid:[0,1,2],consider:[0,1,2],consideration_queu:[0,1,2],consideration_queue_indic:1,consideration_set:[0,1],consideration_set_execut:[0,1,2],consist:[1,2],constitu:1,constitut:1,construct:[0,1],constructor:[0,1],contain:[0,2],context:0,continu:1,conveni:[0,2],converg:0,copi:1,correspond:[0,1,2],count:[0,1],counter:1,cover:1,creat:[2,3],creation:2,cur_consideration_set:1,cur_consideration_set_execut:1,cur_consideration_set_has_chang:1,cur_index:1,cur_nod:1,current:[0,1,2,3],current_tim:2,custom:3,custom_parameter_gett:0,custom_parameter_valid:0,cycl:1,d:3,dag:3,data:[0,1],decim:1,def:0,default_absolute_time_unit:1,default_execution_id:1,defin:[0,1,2],delta:0,depend:[0,1,3],depth:1,descript:1,desir:[0,1],detect:1,determin:[0,1],dict:[0,1,2],dictionari:[1,2,3],differ:[0,1],digraph:[1,3],direct:[0,3],directli:1,divis:[0,2],document:3,doe:1,done:0,doubt:0,drawback:1,due:1,durat:1,dure:[0,2],dynam:0,e:[0,1,2],each:[0,1,2],earlier:1,edg:1,effect:0,either:0,element:0,elig:1,empti:1,enable_current_tim:2,end:[0,1,2],end_environment_sequ:[0,1],end_inclus:0,ensur:0,entri:[0,1,2],environ:[0,1,2],environment_sequ:[0,1,2],environment_state_upd:[0,1,2],environmentstateupd:0,epsilon:0,equal:0,equival:0,error:1,es:[0,1],etc:0,evalu:[0,1],evenli:0,everi:[0,1,2],everyncal:[0,1,3],everynpass:[0,1],exact:0,exact_tim:1,exact_time_mod:0,exactli:[0,1],exampl:[0,1],except:0,execut:[2,3],execution_id:[0,1],execution_list:1,execution_sequ:1,exist:[0,1,2],expect:0,explicitli:[0,1],express:3,fals:[0,1,2],finer:2,finest:2,finish:0,first:[0,1],fix:[0,1],flexibl:0,follow:[0,1,2],form:[0,1],formal:0,format:[2,3],forth:1,fraction:1,fragment:0,frequenc:0,from:[1,2,3],fter:0,full:[1,2],func:0,further:3,g:[0,1,2],gap:1,gener:[0,1,3],get:1,get_child:2,get_par:2,get_time_by_time_scal:2,get_total_times_rel:2,github:3,give:2,given:[0,1],go:3,govern:[0,1],grain:2,granular:2,graph:[0,1],graph_schedul:[0,1,2,3],guarante:1,ha:[0,1,2],handl:1,hashabl:1,have:[0,1,2],here:[0,1,3],histori:[1,2],how:[0,1],howev:1,http:3,i:[0,1,2],ident:0,identifi:1,immedi:0,implement:[0,1],implicit:1,inclin:0,includ:0,inclus:0,increas:2,increasingli:2,increment:[1,2],increment_tim:2,independ:[0,1],index:[0,2,3],indic:[0,1],individu:1,inexact:1,infer:1,info:0,initi:[0,1],instanc:2,instanti:0,instead:[0,1],intact:1,interv:[0,2],involv:1,io:3,irrelev:1,is_finish:0,is_satisfi:0,issu:1,item:0,iter:[0,1,2],its:[0,1,2],itself:0,justran:0,keep:2,kei:[0,1],keyword:0,kmantel:3,kwarg:[0,1],larg:2,last:[0,1,2],later:1,latest:2,least:[0,1,2],len:1,length:1,less:0,level:2,life:2,like:1,limit:1,linear:1,link:3,list:[1,2,3],logic:1,lower:2,mai:[0,1,2,3],main:3,maintain:[0,2],make:[0,1],manag:[0,2],mani:[0,1],map:[0,1,3],max_depth:2,maximum:0,measur:2,member:[0,1],memori:2,met:1,method:[0,1],microsecond:1,millisecond:[0,1,2],mode:0,model:1,modul:3,more:[0,1,2],morev:1,most:[0,1],multipl:[0,1],must:[0,1],my_schedul:0,n:0,name:[0,2],necessari:0,need:1,networkx:[1,3],never:[0,1],next:[0,1],node:[0,1,2,3],node_a:0,node_b:0,node_list:0,non:1,none:[0,1,2],normal:1,note:0,notimpl:1,nuclear:2,number:[0,2],nwhen:0,object:[1,2],occur:[0,1,2],onc:[0,1,2],one:[0,1,2],onli:[0,1,2],open:[1,2],oper:0,opposit:0,option:[0,1,3],order:[1,2,3],origin:1,other:[0,1],otherwis:[0,1],over:[0,1,2],overview:3,overwrit:[0,1],overwritten:[0,1],own:[0,1],owner:[0,1],packag:0,page:3,paramet:[0,1,2],parametr:0,parent:[0,1,2],particular:1,pass:[0,1,2],pass_:2,pattern:[1,3],per:1,phase:1,pint:[0,1,2,3],pip:3,pleas:1,point:[0,1],posit:[0,1],possibl:1,precis:[1,2],predefin:3,previou:0,previous:[1,2],previous_tim:2,print:3,prior:3,problem:1,process:1,produc:1,project:1,properti:[0,2],provid:[0,1,2,3],pseudocod:1,pypi:3,python:1,quantiti:[0,1,2],queri:2,query_time_scal:2,queue:[0,1],rang:0,rather:[1,2],reach:0,real:[1,3],receiv:1,recurr:1,refer:[2,3],rel:0,relat:2,releas:3,remain:1,remov:2,remove_time_scale_alia:2,repeat:0,report:0,repres:2,requir:0,reset:[0,1,2],respect:[0,1],respons:0,restrict:3,result:[0,1],root:2,rtol:0,run:[0,1,2,3],s:[0,1,2],same:1,satisfact:0,satisfi:[0,1],scalar:0,scale:2,sched:3,schedul:[0,2],schedulingmod:1,scope:2,script:0,search:3,second:[0,1],section:1,see:[0,1],self:2,sequenc:[1,3],sequenti:[0,1],seri:0,set:[0,1,2,3],set_time_scale_alia:2,settl:0,should:[0,1,2],signal:1,simpl:[1,2],simple_tim:2,simplest:1,simpletim:2,simpli:0,simplifi:2,simul:2,simultan:[1,2],sinc:[0,1,2],singl:[0,1,2],situat:1,six:0,skip_environment_state_update_time_incr:1,smaller:[1,2],so:[0,1],sole:0,solveabl:1,some:1,specif:[0,1,3],specifi:[1,2,3],standard:[0,1],start:[0,1],start_inclus:0,state:[0,1],statu:0,store:[0,2],str:2,string:0,structur:[1,3],subclass:0,subject:1,subsequ:1,subset:[1,2],subtre:2,suppli:[0,1],support:[0,1,3],system:1,t:[0,1],tag:3,tag_nam:3,take:0,target:2,termin:0,termination_cond:1,test:0,th:2,than:[0,1,2],thei:[0,1],them:1,themselv:0,thereaft:0,therefor:0,thi:[0,1,2],third:1,those:0,though:1,thresh:0,threshold:0,through:[0,1,2],thu:0,tick:2,time:[0,3],time_ref:2,time_scal:[0,1,2],timehistorytre:2,timeinterv:[0,1],timesc:0,timescal:[0,1,2],timetermin:[0,1],toler:0,topolog:3,toposort:1,total:[0,2],total_tim:2,track:2,transcend:0,tree:2,trigger:1,two:[0,1],type:[0,1,2],typic:2,under:[0,1],unexpect:[0,1],unexpectedli:1,uniqu:1,unit:[0,1,2,3],unord:1,unpack:0,unsatisfact:0,until:[0,1],up:2,updat:[0,1],upon:0,us:[0,1,2,3],useabl:0,user:[0,3],val:0,valu:[0,1,2],vari:0,varieti:0,variou:2,wa:0,wai:1,wait:[0,1],want:[0,1],were:1,what:2,when:[0,1],whenev:[0,1],whenfinish:0,whenfinishedal:0,whenfinishedani:0,where:[0,1],whether:[0,1,2],which:[0,1,2,3],whilenot:0,whose:[0,1,2],wider:2,within:[0,1,2],without:0,wrapper:2,x:0,yield:1,you:[0,1,2],zero:[0,1]},titles:["Condition","Scheduler","Time","Graph Scheduler"],titleterms:{"class":[0,1],absolut:1,algorithm:1,condit:[0,1],content:3,creat:[0,1],custom:0,exact:1,exampl:3,execut:[0,1],graph:3,indic:3,instal:3,list:0,mode:1,overview:[0,1,2],pre:0,refer:[0,1],schedul:[1,3],specifi:0,structur:0,tabl:3,termin:1,time:[1,2]}})