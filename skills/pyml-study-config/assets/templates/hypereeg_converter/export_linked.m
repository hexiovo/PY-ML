function export_linked(configPath)
% Decode native MATLAB tables only. Never reshape numeric tensors.
c=jsondecode(fileread(configPath)); addpath(c.memory_helper_root);
p=load(fullfile(c.input,'特征结果索引.mat'));
log=fopen(fullfile(c.output,'native_manifest.jsonl'),'w','n','UTF-8');
cleanup=onCleanup(@() fclose(log));
counts=struct('exported_tables',0,'unmapped_tables',0,'outside_profile',0,'failed_artifacts',0);
indexes=cell(height(p.PackageIndex),1);pathLists=cell(height(p.PackageIndex),1);totals=zeros(height(p.PackageIndex),1);
for n=1:height(p.PackageIndex)
 q=load(fullfile(c.out_root,string(p.PackageIndex.relativePath(n))),'FeatureExtractionIndex');
 indexes{n}=q.FeatureExtractionIndex;
 list=unique(string(indexes{n}.features.payload_path),'stable');list=list(strlength(list)>0);
 pathLists{n}=list;totals(n)=numel(list);
end
c.native_totals=totals;clear q list;
for k=1:height(p.PackageIndex)
 indexPath=fullfile(c.out_root,string(p.PackageIndex.relativePath(k)));
 x=indexes{k};
 taskDir=fullfile(c.output,sprintf('task_%02d',k));mkdir(taskDir);
 writetable(x.features,fullfile(taskDir,'feature_status.csv'),'Encoding','UTF-8');
 writetable(x.artifactIndex,fullfile(taskDir,'artifact_index.csv'),'Encoding','UTF-8');
 paths=pathLists{k};
 for j=1:numel(paths)
  clear d a value;
  path=paths(j); r=struct('task',k,'source',path,'status','pending');
  if mod(j-1,10)==0,writeProgress(c,k,height(p.PackageIndex),j-1,numel(paths),path,counts);end
  featureRows=x.features(string(x.features.payload_path)==path,:);
  featureIDs=string(featureRows.feature_id);
  if ~any(startsWith(featureIDs,["signal.","spectrum.","nonlinear."]))
   r.status='outside_scalar_profile';r.feature_ids=featureIDs;
   r.reason='Current profile converts signal/spectrum/nonlinear scalar tables only; other families remain in native artifacts without inferred tensor axes.';
   counts.outside_profile=counts.outside_profile+1;
   fprintf(log,'%s\n',jsonencode(r));
   if mod(j,10)==0,writeProgress(c,k,height(p.PackageIndex),j,numel(paths),path,counts);end
   continue
  end
  try
   [estimated,method]=HyperEEG.MultiCH.core.FeatureArtifactMemoryEstimate(path,"Artifact");
   [~,systemMemory]=memory;
   available=double(systemMemory.PhysicalMemory.Available);
   budget=min(c.max_artifact_load_gib*1024^3,max(0,available-c.reserve_free_gib*1024^3));
   if estimated>budget
    error('PYML:ResourceLimit','MAT requires estimated %.1f MiB; available admission budget %.1f MiB. No payload loaded.',estimated/1024^2,budget/1024^2);
   end
   r.memory=struct('estimated_bytes',estimated,'method',method,'budget_bytes',budget,'available_physical_bytes',available);
   d=load(path,'Artifact');a=d.Artifact;
   r.artifact_id=a.artifactId;r.producer=a.producerId;r.quality=a.quality;
   r.source_ids=a.sourceIds;r.scope=a.scope;
   if isfield(a.audit,'sourcePath'),r.source_path=a.audit.sourcePath;end
   tables=findTables(a.payload,"Artifact.payload"); r.tables={};
   for z=1:numel(tables)
    value=a.payload;parts=split(erase(tables(z),"Artifact.payload."),'.');
    if tables(z)~="Artifact.payload",for part=parts',value=value.(part);end;end
    columns=string(value.Properties.VariableNames);
    recognized=ismember(columns,string(c.metric_names)) | startsWith(columns,string(c.metric_prefixes));
    if ~ismember('channel_index',columns) || ~any(recognized) || a.scope~="record" || numel(a.sourceIds)~=1
     r.tables{end+1}=struct('path',tables(z),'status','unsupported_mapping','reason','No verified scalar record mapping in current plugin','rows',height(value),'columns',{value.Properties.VariableNames});
     continue
    end
    dest=fullfile(taskDir,sprintf('table_%05d_%02d',j,z));mkdir(dest);
    req=struct('input',path,'output',dest,'table_path',tables(z));
    reqPath=fullfile(dest,'request.json');writeJSON(reqPath,req);
    tr=struct('path',tables(z),'directory',dest,'status','pending');
    try
     export_loaded_table(req,struct('Artifact',a));tr.status='exported';
    catch e
     if ismember(string(e.identifier),["PYML:UnsupportedColumn","PYML:NotTable"])
      tr.status='unsupported';tr.reason=e.message;
     else,rethrow(e);end
    end
    r.tables{end+1}=tr;
   end
   if isempty(tables),r.status='unsupported_tensor_or_no_table';else,r.status='inspected';end
  catch e
   r.status='failed';r.reason=e.message;
   counts.failed_artifacts=counts.failed_artifacts+1;
   fprintf(log,'%s\n',jsonencode(r));
   rethrow(e);
  end
  if isfield(r,'tables')
   for tr=r.tables
    if strcmp(tr{1}.status,'exported'),counts.exported_tables=counts.exported_tables+1;
    else,counts.unmapped_tables=counts.unmapped_tables+1;end
   end
  end
  fprintf(log,'%s\n',jsonencode(r));
  if mod(j,10)==0,writeProgress(c,k,height(p.PackageIndex),j,numel(paths),path,counts);end
  if mod(j,100)==0,fprintf('Task %d: %d / %d artifacts\n',k,j,numel(paths));end
 end
 writeProgress(c,k,height(p.PackageIndex),numel(paths),numel(paths),'Task native export finished',counts);
 fprintf('Task %d native export finished: %d artifacts\n',k,numel(paths));
end
end
function writeProgress(c,k,tasks,done,total,current,counts)
value=struct('task',k,'tasks',tasks,'completed',done,'total',total,'current',string(current),'counts',counts);
value.overall_completed=sum(c.native_totals(1:k-1))+done;value.overall_total=sum(c.native_totals);
% Replace-by-rename conflicts with Windows readers' delete sharing mode.
% Overwrite the disposable snapshot; readers retry incomplete JSON.
writeJSON(fullfile(c.output,'native_progress.json'),value);
end
function paths=findTables(v,p)
paths=strings(0,1);
if istable(v),paths=p;
elseif isstruct(v)&&isscalar(v)
 for n=string(fieldnames(v))',paths=[paths;findTables(v.(n),p+"."+n)];end
end
end
function writeJSON(path,v)
f=fopen(path,'w','n','UTF-8');cleanup=onCleanup(@() fclose(f));fprintf(f,'%s',jsonencode(v));
end
