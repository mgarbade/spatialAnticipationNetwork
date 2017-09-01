function options = processInputArgs(options, varargin)



optionNames = fieldnames(options);
nArgs = length(varargin);
if round(nArgs/2) ~= nArgs/2
   error('EXAMPLE needs propertyName/propertyValue pairs')
end
for pair = reshape(varargin,2,[])
   inpName = pair{1};
   if any(strcmp(inpName,optionNames))
      options.(inpName) = pair{2};
   else
      error('%s is not a recognized parameter name',inpName)
   end
end