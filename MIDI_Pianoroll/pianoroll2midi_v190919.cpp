//g++ -O2 -I/Users/eita/boost_1_63_0 -I/Users/eita/Dropbox/Research/Tool/All/ pianoroll2midi_v190919.cpp -o pianoroll2midi
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<cmath>
#include<cassert>
#include<algorithm>
#include "PianoRoll_v170503.hpp"
using namespace std;

int main(int argc, char** argv) {
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc<3){cout<<"Error in usage: >pianoroll2midi in_spr/ipr.txt out.mid track0:71 track2:0 ..."<<endl; return -1;}

	string infile=string(argv[1]);
	string outfile=string(argv[2]);

	if(infile.find("spr.txt")==string::npos && infile.find("ipr.txt")==string::npos){
		cout<<"Input file must be spr.txt or ipr.txt"<<endl;
		return -1;
	}//endif

	vector<int> inputProgramChangeData;
	inputProgramChangeData.assign(16,0);
	for(int k=3;k<argc;k+=1){
		string trackInfo=string(argv[k]);
		string trackNum,programNum;
		if(trackInfo.substr(0,5)!="track"){
			cout<<"Error: Program changes must be specified as: track0:71 track2:0 ..."<<endl; return -1;
		}//endif
		trackInfo=trackInfo.substr(5);
		if(trackInfo.find(":") == string::npos){
			cout<<"Error: Program changes must be specified as: track0:71 track2:0 ..."<<endl; return -1;
		}//endif
		trackNum=trackInfo.substr(0,trackInfo.find(":"));
		programNum=trackInfo.substr(trackInfo.find(":")+1);
		if(atoi(trackNum.c_str())<0 || atoi(trackNum.c_str())>=16
		   || atoi(programNum.c_str())<0 || atoi(programNum.c_str())>=128){
			cout<<"Error: Track number must be 0,...,15. Program change number must be 0,...,127"<<endl; return -1;
		}//endif
		inputProgramChangeData[atoi(trackNum.c_str())]=atoi(programNum.c_str());
	}//endfor k

	Midi midi;
	midi.SetProgramChangeData(inputProgramChangeData);

	PianoRoll pr;

	if(infile.find("spr.txt")!=string::npos){
		pr.ReadFileSpr(infile);
	}else if(infile.find("ipr.txt")!=string::npos){
		pr.ReadFileIpr(infile);
	}else{
cout<<"Error in usage: >pianoroll2midi (0:spr/1:ipr) in_spr/ipr.txt out.mid track0:71 track2:0 ..."<<endl; return -1;
	}//endif

	midi=pr.ToMidi();
	midi.SetStrData();

//  MIDIData midiData;
//  midiData.SetProgramChangeData(inputProgramChangeData);
//  for(int i=0;i<16;i+=1){
//cout<<i<<" "<<midiData.programChangeData[i]<<endl;
//  }//endfor i

//  midiData.ReadIPRData(string(argv[1]));
//  midiData.SetStrData();

	ofstream ofs;
	ofs.open(outfile.c_str(), std::ios::binary);
	ofs<<midi.strData;
	ofs.close();

	return 0;
}//end main
