
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef CMDPARSER_H
#define CMDPARSER_H

#include <string>
#include <vector>
#include <map>

namespace rtrt {
class CMDParser{
 public:
  CMDParser(std::string arguments);
  ~CMDParser();

  void addOpt(std::string opt, int numValues, std::string optlong, std::string help = "");
  void showHelp();
  void init(int& argc, char** argv);

  int isOptSet(std::string opt);

  std::vector<int> getOptsInt(std::string);
  std::vector<float> getOptsFloat(std::string);
  std::vector<std::string> getOptsString(std::string);


  std::vector<std::string> getArgs() const;

 private:
  std::map<std::string,std::vector<std::string>* > _opts;
  std::map<std::string,int> _optsNumValues;
  std::map<std::string,std::string> _optslong;
  std::vector<std::string> _args;
  std::string _help;
  std::string _arguments;
};
}



#endif //#ifndef CMDPARSER_H
