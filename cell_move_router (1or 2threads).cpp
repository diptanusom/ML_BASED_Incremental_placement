// TODO : Make the cell adjacency according to pin connectivity

#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <queue>
#include <chrono>
#include <thread>
#include <mutex>
//#include <boost/python.hpp>
#include "fdeep/fdeep.hpp"
#include <unordered_map>
#include <time.h>
#include "data.hxx"
#define ROUTER_MAXTIME 120
#define PLACER_MAXTIME 120

#define CONNECTIVITY_SCALING_FACTOR 1
#define PIN_CONNECTIVITY_SCALING_FACTOR 10000
#define SUP_DEM_SCALING_FACTOR 10
#define MIN_STEP_VAL 3
#define MAX_STEP_VAL 10

#define RANDOM_FUNC_SEED 65535

#define FIXED_CELL_IDX 1
unsigned trial =0;
using namespace std;
using namespace fdeep;

GlobalVariables gv;
unsigned numRandomPlacementsGeneratedY1 = 0;
unsigned numRandomPlacementsGeneratedY2 = 0;
unsigned numOverflowsInDataSampleCreation = 0;
//const auto model = fdeep::load_model("fdeep_model.json");
unsigned totalPlacementsGenerated = 0;
std::mutex m, m1, m3, cmListmutex;
bool routerLog = false, placerLog = false, steinerPointsLog = false;
//unsigned wirelengthCost;
priority_queue<routeGuide *, vector<routeGuide *>, CompareCellCost> tempWf;

/*PyObject *pModelObj, *pModelPredictFuncArgsObj;
PyObject *pRandomFuncObj;*/

static unsigned routerCalled = 0;
static unsigned routerSuccessForAGivenNet = 0;
static unsigned routerSuccessForAGivenCell = 0;
static unsigned routerFailureForAGivenCell = 0;

static unsigned numPredicitionCalls = 0;
static unsigned numTotalTrials = 0;

int pinCount = 0;
unsigned firstPartitionStart = 0;
unsigned firstPartitionEnd = 0;
unsigned secondPartitionStart = 0;
unsigned secondPartitionEnd = 0;
unsigned thirdPartitionStart = 0;
unsigned thirdPartitionEnd = 0;
unsigned fourthPartitionStart = 0;
unsigned fourthPartitionEnd = 0;
unsigned fifthPartitionStart = 0;
unsigned fifthPartitionEnd = 0;
unsigned sixthPartitionStart = 0;
unsigned sixthPartitionEnd = 0;
unsigned seventhPartitionStart = 0;
unsigned seventhPartitionEnd = 0;
unsigned eighthPartitionStart = 0;
unsigned eighthPartitionEnd = 0;

unsigned maxNetsPerCell = 0;
void clearNumPinsArray();
unsigned whichPartition(unsigned row);
unsigned partitionMoveCell(vector<Inst *> instList, unsigned rowBegIdx,
						   unsigned rowEndIdx, unsigned colBegIdx, unsigned colEndIdx,
						   unsigned &numCellsMoved, double maxTime, ofstream &logs);

bool checkIfCellMovable(Inst *Cell, ofstream &logs);
void totalSupply();
unsigned netWirelengthDemand(Net *net, bool cal_demand, bool deleteSegment);
unsigned totalNetWirelengthDemand();
void getBlockDemand();
unsigned getNoCellInGridCount(MasterCell *cellName,
							  vector<MasterCell *> cellList);
unsigned getNoCellInGridCount(MasterCell *cellName1, MasterCell *cellName2,
							  vector<MasterCell *> cellList, unsigned &count1, unsigned &count2);
void getCellExtraDemand();
int findStepSizeAboveAndBelow(int number, unsigned min, unsigned max);
bool isPrime(unsigned number);
int findMultipleGreaterThanOneAbove(unsigned number, unsigned min);
void getDemand();
float getCongestion(unsigned x, unsigned y, unsigned z);
int getOverFlow(unsigned x, unsigned y, unsigned z);
float mean(float *X, int N);
float standardDeviation(float *X, int N);
void standardScaler(float *X, unsigned N);
int parse(const string &ifname, ofstream &logs);
int dumpOutput(ostream &os);
unsigned printRandoms(int lower, int upper);
int generateDemandSuppVals();
void getGridCongestion();
float getCellCongestion(Inst *);
unsigned findPinIndexForWindow(Pin *pin);

int partitionDesign(ofstream &dataFile, unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx);
int partitionDesign(vector<Inst *> &instList, ofstream &dataFile, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex, unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
					list<Inst *> &cellsInWindow, list<Inst *> &cellsInRowWindow, list<Inst *> &cellsBeyondRowWindow);
unsigned generateBitVectorOfCongestionMatrix(unsigned row, unsigned col);
void generateDataSample(ofstream &dataFile, unsigned movableCellIndex,
						unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx, unsigned compareDem, bool initialize, bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL],
						unsigned *rowPosCell, unsigned *colPosCell, unsigned *cellIdxList, unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k, float *X, float *Y);
void generateOutputDataSample(ofstream &dataFile, unsigned rowStartIndex, unsigned rowEndIndex, unsigned colStartIndex, unsigned colEndIndex,
							  unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
							  bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL], unsigned *rowPosCell, unsigned *colPosCell,
							  unsigned *cellIdxList, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, list<Inst *> &movableCellList, unsigned k);
int slidingWindowScan(vector<Inst *> &instList, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex, unsigned rowSize, unsigned colSize, unsigned rowStep, unsigned colStep,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials ,const fdeep::model &model); // rowSize and colSize are sliding window sizes
																																																					 // rowStartIndex, colStartIndex, rowEndIndex, colEndIndex define the size of the particular partition to run sliding window upon.
																																																					 // rowStep and colStep have thier usual meaning
void clearCellArrays(unsigned *cellIdxList, unsigned *rowPosCell, unsigned *colPosCell, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList);					 //  To be changed for number of pins above 50 in the design
bool isCellPresentInCellIdxList(unsigned cellIdx, unsigned *cellIdxList);

int pythonPredictionCode(float* X, float *Y,const auto &model);

unsigned commitRouteForCell(Inst *Cell, bool commit, vector<Net *>::iterator itr,
							ofstream &logs);
bool findRoutingForCell(Inst *Cell, vector<Net *>::iterator &itr,
						unsigned rowBegIdx, unsigned rowEndIdx, unsigned colBegIdx,
						unsigned colEndIdx, ofstream &logs);
int createPinConnectivityForDesign(bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL], unsigned *cellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k);
void cellExtraDemand(unsigned x, unsigned y, bool removeDemand);
void movedCellExtraDemand(Inst *Cell, unsigned row, unsigned col);
void movedCellBlockDemand(Inst *Cell, unsigned row, unsigned col);
unsigned wirelengthForCell(Inst *Cell, ofstream &logs);
int deltahpwlForCell(Inst *CelltoMove, unsigned xpos, unsigned ypos,
					 unsigned &newtotalhpwl, ofstream &logs);
unsigned hpwlForCell(Inst *Cell);
unsigned pMoveCell(Inst *Cell, unsigned rowBegIdx, unsigned rowEndIdx,
				   unsigned colBegIdx, unsigned colEndIdx,
				   unsigned mCellGridRow, unsigned mCellGridCol, ofstream &logs,unsigned &numCellsMoved);
unsigned commitRoute(Net *N, bool commit, ofstream &logs);
bool findRoute(Net *N, unsigned rowStart, unsigned rowEnd, unsigned colStart,
			   unsigned colEnd, ofstream &logs);
bool getNbforGivenCell(unsigned rowStart, unsigned rowEnd, unsigned colStart,
					   unsigned colEnd, routeGuide *cell, routeGuide *destination, Net *N,
					   unsigned minLayer,
					   priority_queue<routeGuide *, vector<routeGuide *>, CompareCellCost> &tempWf,
					   unsigned &wirelengthCost, ofstream &logs);
int isEqual(routeGuide *x, routeGuide *y);
int backtrack(routeGuide *cell, Net *N, ofstream &logs);
void initializeRouteGrid(unsigned rowStart, unsigned rowEnd, unsigned colStart,
						 unsigned colEnd);
void initializeRoute(unsigned rowStart, unsigned rowEnd, unsigned colStart,
					 unsigned colEnd);
unsigned getBoundingBoxCost(routeGuide *source, routeGuide *destination);
unsigned getDemandsOfCell(Inst *Cell);
int pythonRandomPrediction(float *);

float getStepForWindow(float num, float near);
void clearPinConnectivityMatrix();
void clearPinConnectivityMatrix(bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL]);
ofstream predictionLogs;
void dataFileDemandSupplyHeadersCreation(ofstream &predictionLogs);

// NEW FUNCTIONS FOR NET BASED IMPLEMENTATION:-
Inst *returnInstFromMovableCellList(unsigned cellIdx, list<Inst *> &movableCellList);
unsigned setResetInternalNetindex(Inst *cellOne, std::unordered_set<string> &pinNetList, bool set, bool reset);
unsigned createNetBasedConnectivityMatrix(unsigned numCells, unsigned *cellIdxList, unsigned net_connectivity_matrix[][MAX_NETS + 1], list<Inst *> &movableCellList);
void clearNetConnectivityMatrix(unsigned *net_connectivity_matrix[][MAX_NETS + 1]);
void generateDataSampleCurrentCell(ofstream &dataFile, unsigned movableCellIndex, unsigned net_connectivity_matrix[][MAX_NETS + 1],
								   unsigned numNetsCurrCell, unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx,
								   unsigned endGridColIdx, unsigned *rowPosCell, unsigned *colPosCell, unsigned *cellIdxList, unsigned *demandValCell,
								   unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k, float *X, float *Y,
								   bool initialize);
int generateOutputDataSampleCurrentCell(ofstream &dataFile, unsigned rowStartIndex, unsigned rowEndIndex, unsigned colStartIndex, unsigned colEndIndex,
										 unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
										 unsigned net_connectivity_matrix[][MAX_NETS + 1], unsigned *rowPosCell,
										 unsigned *colPosCell, unsigned *cellIdxList, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray,
										 pinIndexes *pinIndexList, list<Inst *> &movableCellList, unsigned k, unsigned numNetsCurrCell,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials,const fdeep::model &model);
int partitionDesignClustered(vector<Inst *> &instList, ofstream &dataFile, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex,
							 unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
							 list<Inst *> &cellsInWindow, list<Inst *> &cellsInRowWindow, list<Inst *> &cellsBeyondRowWindow,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials,const fdeep::model &model);
unsigned returnL1Blockage(Inst *cell);
void produceRouterLogs();
unsigned numRouterFailiures = 0;
unsigned numPredictionFailiures = 0;
unsigned numPredictionFailiuresWirelength = 0;
unsigned numNotFoundBetterSolution = 0;
unsigned numRouterSuccess = 0;
unsigned numRouteOverflowFailiures = 0;
time_t start;
unsigned initialHPWLZero = 0;
unsigned numRepeat = 0;
string outFile;
unsigned maxPins = 0;
unsigned threadcount=0;
unsigned timeLimit =0;
bool localModel=false;
unsigned FLAG =0;
int main(int ac, char **av)
{
	ofstream logs;
	predictionLogs.open("predictions_logs_new.csv");
	dataFileDemandSupplyHeadersCreation(predictionLogs);
	unsigned numNetsCovered = 0;

	logs.open("logs_new.txt");

	auto timebefore = chrono::system_clock::to_time_t(chrono::system_clock::now());
	logs << ctime(&timebefore) << endl;

	unsigned int totalHpwl = 0, newTotalHpwl = 0;
	vector<Inst *>::iterator itr;

	logs << "argc: " << ac << endl;

	if (ac < 3)
	{
		logs << "Give input file name: " << ac << endl;
		logs << "Usage: ./cell_move_router <input.txt> <output.txt>" << endl;
		logs << "Debugging options: -parser -hpwl";
		logs << "-router -placer -steinerPoints" << endl;
		return 1;
	}

	bool parserLog = false, hpwlLog = false;

	if (ac > 3)
	{

		for (int i = 4; i < ac; i++)
		{
			string option(av[i]);
			if (option == "-parser")
			{
				parserLog = true;
			}

			else if (option == "-hpwl")
			{
				hpwlLog = true;
			}

			else if (option == "-router")
			{
				routerLog = true;
			}

			else if (option == "-placer")
			{
				placerLog = true;
			}

			else if (option == "-steinerPoints")
			{
				steinerPointsLog = true;
			}
			else if (option == "-thread=0")
			{
				threadcount=0;
			}
			else if (option == "-thread=2")
			{
				threadcount=2;
			}
			else if (option == "-thread=4")
			{
				threadcount=4;
			}
			else if (option == "-thread=8")
			{
				threadcount=8;
			}
			else if(option == "-localModel=true"){
				localModel =true;
			}
			else if(option == "-localModel=false"){
				localModel =false;
			}
			else if(option == "-commentFunction=true"){
				FLAG = 0;
			}else if(option == "-commentFunction=false"){
				FLAG=1;
			}
			else
			{
				logs << "-Invalid option selected" << endl;
				logs << "Available options: -parser -hpwl";
				logs << " -router -placer -steinerPoints" << endl;
				return 1;
			}
		}
	}

	string inputFileName(av[1]);
	string outputFileName(av[2]);
	outFile = outputFileName;
	timeLimit = stoi(av[3]);

	logs << "Parsing: " << inputFileName << endl;
	parse(inputFileName, logs);
	// Ignore this for now (Put for Multithreading)
	logs << "P1 size : " << gv.dd.firstPartitionNetList.size() << endl;
	logs << "P2 size : " << gv.dd.secondPartitionNetList.size() << endl;
	/*logs << "P3 size : " << gv.dd.thirdPartitionNetList.size() << endl;
	logs << "P4 size : " << gv.dd.fourthPartitionNetList.size() << endl;
	logs << "P5 size : " << gv.dd.fifthPartitionNetList.size() << endl;
	logs << "P6 size : " << gv.dd.sixthPartitionNetList.size() << endl;
	logs << "P7 size : " << gv.dd.seventhPartitionNetList.size() << endl;
	logs << "P8 size : " << gv.dd.eighthPartitionNetList.size() << endl;
	logs << "fixed net size : " << gv.dd.fixedNetList.size() << endl;*/
	

	cout << "Row end :: " << gv.rowEndIdx << endl;
	cout << "Col end :: " << gv.colEndIdx << endl;
	
	unsigned rowStep = findStepSizeAboveAndBelow(gv.rowEndIdx, MIN_STEP_VAL, MAX_STEP_VAL);// This is incorrect
	// Needs to be changed
	unsigned colStep = findStepSizeAboveAndBelow(gv.colEndIdx, MIN_STEP_VAL, MAX_STEP_VAL);
	logs << "Row step calculated to be : " << rowStep << endl;
	logs << "Col step calculated to be : " << colStep << endl;
	if (parserLog)
	{
		ofstream parseOut;
		parseOut.open("parse.txt");
		dumpOutput(parseOut);
		parseOut.close();
		logs << "Generated parse output file: parse.txt" << endl;
	}


	const auto model1 = fdeep::load_model("fdeep_model.json");
	//const auto model2 = fdeep::load_model("fdeep_model.json");

	totalSupply();		 // ? <-------------Def + nonDefSup calculated
	getDemand();		 // ! !!!!!!!!!!!!!!!!!!!!!!!!!DOUBT
	getGridCongestion(); // ?<------------------- Calculating demand/supply and deciding congestion based unpon a congestionn threshold
	// getGridCongestioni s obselete
	cout << "Supply, Demand and Congestion Calculated" << endl;
	ofstream logs0,logs1,logs2,logs3,logs4,logs5,logs6,logs7;
	logs << "INITIAL HPWL: " << endl;
	totalHpwl = gv.dd.totalHpwl(logs); // Calculating the total initial wirelength
	logs << "Total HPWL is " << totalHpwl << endl;
	
	unsigned numCellsMovedFirstPartition = 0, numCellsMovedSecondPartition = 0,numCellsMovedThirdPartition = 0, numCellsMovedFourthPartition = 0,
			 numCellsMovedFifthPartition = 0, numCellsMovedSixthPartition = 0,
			 numCellsMovedSeventhPartition = 0, numCellsMovedEighthPartition = 0;
	unsigned numCellsMoved =0;
	unsigned numPredictionTrialsFirstPartition =0,numPredictionTrialsSecondPartition =0,numPredictionTrialsThirdPartition =0,numPredictionTrialsFourthPartition =0,numPredictionTrialsFifthPartition =0,numPredictionTrialsSixthPartition =0,numPredictionTrialsSeventhPartition =0,numPredictionTrialsEighthPartition =0;
	unsigned numPredictionTrials=0;

	
	initializeRouteGrid(gv.rowBeginIdx, gv.rowEndIdx, gv.colBeginIdx, gv.colEndIdx);
	time(&start);
	if (threadcount == 0) {
		slidingWindowScan(gv.dd.instList,gv.rowBeginIdx,gv.colBeginIdx,gv.rowEndIdx,gv.colEndIdx,10,10,rowStep,colStep,logs,numCellsMoved, numPredictionTrials,model1);
		cout << "numPredictionTrials in zero partition = " << numPredictionTrials << endl;
		cout << "numCell moved = " << numCellsMoved << endl;
	} else if (threadcount == 2) {
		logs0.open("logs0.txt");
		logs1.open("logs1.txt");
		std::thread tm1(slidingWindowScan, std::ref(gv.dd.instListFirstPartition), firstPartitionStart, gv.colBeginIdx, firstPartitionEnd,
				gv.colEndIdx, 10 ,10,rowStep,colStep ,std::ref(logs0),std::ref(numCellsMovedFirstPartition),std::ref(numPredictionTrialsFirstPartition),std::reference_wrapper<const fdeep::model>(model1));
		std::thread tm2(slidingWindowScan, std::ref(gv.dd.instListSecondPartition),secondPartitionStart,  gv.colBeginIdx, secondPartitionEnd,
				gv.colEndIdx, 10, 10,rowStep ,colStep,std::ref(logs1),std::ref(numCellsMovedSecondPartition),std::ref(numPredictionTrialsSecondPartition),std::reference_wrapper<const fdeep::model>(model1));
		tm1.join();
		tm2.join();
		logs0.close();
		logs1.close();
		cout << "number of cells moved in first partition =" << numCellsMovedFirstPartition << endl;
		cout << "number of cells moved in second partition =" << numCellsMovedSecondPartition << endl;
		cout << "numPredictionTrials in first partition = " << numPredictionTrialsFirstPartition << endl;
		cout << "numPredictionTrials in second partition = " << numPredictionTrialsSecondPartition << endl;
		cout << "numPredictionTrials = " << numPredictionTrialsSecondPartition + numPredictionTrialsFirstPartition/* + numPredictionTrialsThirdPartition +numPredictionTrialsFourthPartition +numPredictionTrialsFifthPartition+numPredictionTrialsSixthPartition+numPredictionTrialsSeventhPartition+numPredictionTrialsEighthPartition*/<< endl;
        cout << "numCell moved = " << numCellsMovedFirstPartition + numCellsMovedSecondPartition << endl ;
	
	}// }else if(threadcount == 4){
	// 	logs0.open("logs0.txt");
	// 	logs1.open("logs1.txt");
	// 	logs2.open("logs2.txt");
	// 	logs3.open("logs3.txt");
	// 	std::thread tm1(slidingWindowScan, std::ref(gv.dd.instListFirstPartition), firstPartitionStart, gv.colBeginIdx, firstPartitionEnd,
	// 			gv.colEndIdx, 10 ,10,rowStep,colStep ,std::ref(logs0),std::ref(numCellsMovedFirstPartition),std::ref(numPredictionTrialsFirstPartition));
	// 	std::thread tm2(slidingWindowScan, std::ref(gv.dd.instListSecondPartition),
	// 			secondPartitionStart,  gv.colBeginIdx, secondPartitionEnd,
	// 			gv.colEndIdx, 10, 10,rowStep ,colStep,std::ref(logs1),std::ref(numCellsMovedSecondPartition),std::ref(numPredictionTrialsSecondPartition));
	// 	std::thread tm3(slidingWindowScan, std::ref(gv.dd.instListThirdPartition),
	// 		thirdPartitionStart,  gv.colBeginIdx, thirdPartitionEnd,
	// 		gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs2),std::ref(numCellsMovedThirdPartition),std::ref(numPredictionTrialsThirdPartition));
		
	// 	std::thread tm4(slidingWindowScan, std::ref(gv.dd.instListFourthPartition),
	// 		fourthPartitionStart,  gv.colBeginIdx, fourthPartitionEnd,
	// 		gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs3),std::ref(numCellsMovedFourthPartition),std::ref(numPredictionTrialsFourthPartition));
		
	// 	tm1.join();
	// 	tm2.join();
	// 	tm3.join();
	// 	tm4.join();
	// 	logs0.close();
	// 	logs1.close();
	// 	logs2.close();
	// 	logs3.close();
	// 	cout << "number of cells moved in first partition" << numCellsMovedFirstPartition << endl;
	// 	cout << "number of cells moved in second partition" << numCellsMovedSecondPartition << endl;
	// 	cout << "number of cells moved in third partition" << numCellsMovedThirdPartition << endl;
	// 	cout << "number of cells moved in fourth partition" << numCellsMovedFourthPartition << endl;
	// 	cout << "numPredictionTrials in first partition = " << numPredictionTrialsFirstPartition << endl;
	// 	cout << "numPredictionTrials in second partition = " << numPredictionTrialsSecondPartition << endl;
	// 	cout << "numPredictionTrials in third partition = " << numPredictionTrialsThirdPartition << endl;
	// 	cout << "numPredictionTrials in fourth partition = " << numPredictionTrialsFourthPartition << endl;
	// 	cout << "numPredictionTrials = " << numPredictionTrialsSecondPartition + numPredictionTrialsFirstPartition + numPredictionTrialsThirdPartition +numPredictionTrialsFourthPartition /*+numPredictionTrialsFifthPartition+numPredictionTrialsSixthPartition+numPredictionTrialsSeventhPartition+numPredictionTrialsEighthPartition*/<< endl;
    //tm2.join();
	/*std::thread tm3(slidingWindowScan, std::ref(gv.dd.instListThirdPartition),
			thirdPartitionStart,  gv.colBeginIdx, thirdPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs2),std::ref(numCellsMovedThirdPartition),std::ref(numPredictionTrialsThirdPartition));
	//tm3.join();
	std::thread tm4(slidingWindowScan, std::ref(gv.dd.instListFourthPartition),
			fourthPartitionStart,  gv.colBeginIdx, fourthPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs3),std::ref(numCellsMovedFourthPartition),std::ref(numPredictionTrialsFourthPartition));
	//tm4.join();
	std::thread tm5(slidingWindowScan, std::ref(gv.dd.instListFifthPartition),
			fifthPartitionStart,  gv.colBeginIdx, fifthPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs4),std::ref(numCellsMovedFifthPartition),std::ref(numPredictionTrialsFifthPartition));
	//tm5.join();
	std::thread tm6(slidingWindowScan, std::ref(gv.dd.instListSixthPartition),
			sixthPartitionStart,  gv.colBeginIdx, sixthPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs5),std::ref(numCellsMovedSixthPartition),std::ref(numPredictionTrialsSixthPartition));
	//tm6.join();
	std::thread tm7(slidingWindowScan, std::ref(gv.dd.instListSeventhPartition),
			seventhPartitionStart,  gv.colBeginIdx, seventhPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs6),std::ref(numCellsMovedSeventhPartition),std::ref(numPredictionTrialsSeventhPartition));
	//tm7.join();
	std::thread tm8(slidingWindowScan, std::ref(gv.dd.instListEighthPartition),
			eighthPartitionStart,  gv.colBeginIdx, eighthPartitionEnd,
			gv.colEndIdx, 10, 10, rowStep, colStep,std::ref(logs7),std::ref(numCellsMovedEighthPartition),std::ref(numPredictionTrialsEighthPartition));*/
	//tm8.join();
	//tm3.join();
	//tm4.join();
	/*tm5.join();
	tm6.join();
	tm7.join();
	tm8.join();*/

	//logs2.close();
	//logs3.close();
	/*logs4.close();
	logs5.close();
	logs6.close();
	logs7.close();*/
	
	

	// logs << "Number of times router is called = " << routerCalled << endl;

	// logs << "New HPWL: " << endl;
	// newTotalHpwl = gv.dd.totalHpwl(logs);
	// logs << "New Total HPWL is " << newTotalHpwl << endl;
	// logs << "New - Old HPWL is " << (int)(newTotalHpwl - totalHpwl) << endl;
	// logs << "New - Old HPWL = " << newTotalHpwl << " - " << totalHpwl << " = "
	// 	 << (int)(newTotalHpwl - totalHpwl) << endl;

	// logs << "Number of times router Called (in SA) " << routerCalled << endl;
	// logs << "Max nets per cell :" << maxNetsPerCell << endl;
	// cout << "Max Nets per cell  : " << maxNetsPerCell << endl;
	//logs << "Generating output file: " << outputFileName << endl;
	//cout << "number of cells moved in third partition" << numCellsMovedThirdPartition << endl;
	//cout << "number of cells moved in fourth partition" << numCellsMovedFourthPartition << endl;
	/*cout << "number of cells moved in fifth partition" << numCellsMovedFifthPartition << endl;
	cout << "number of cells moved in sixth partition" << numCellsMovedSixthPartition << endl;
	cout << "number of cells moved in seventh partition" << numCellsMovedSeventhPartition << endl;
	cout << "number of cells moved in eighth partition" << numCellsMovedEighthPartition << endl;*/
	//cout << "numMovedCells = " << numCellsMovedFirstPartition + numCellsMovedSecondPartition /*+ numCellsMovedThirdPartition + numCellsMovedFourthPartition + numCellsMovedFifthPartition+numCellsMovedSixthPartition+numCellsMovedSeventhPartition+numCellsMovedEighthPartition*/<< endl;
	/*cout << "numPredictionTrials in third partition = " << numPredictionTrialsThirdPartition << endl;
	cout << "numPredictionTrials in fourth partition = " << numPredictionTrialsFourthPartition << endl;
	cout << "numPredictionTrials in fifth partition = " << numPredictionTrialsFifthPartition << endl;
    cout << "numPredictionTrials in fourth partition = " << numPredictionTrialsSixthPartition << endl;
	cout << "numPredictionTrials in fourth partition = " << numPredictionTrialsSeventhPartition << endl;
	cout << "numPredictionTrials in fourth partition = " << numPredictionTrialsEighthPartition << endl;*/
	// cout << "talco =" << trial << endl;
	 produceRouterLogs();
	 gv.dd.produceOutput(outputFileName);

	// auto timeafter = chrono::system_clock::to_time_t(
	// 	chrono::system_clock::now());

	// logs << ctime(&timeafter) << endl;
	// logs << "Random called for Y1 : " << numRandomPlacementsGeneratedY1 << endl;
	// logs << "Random called for Y2 : " << numRandomPlacementsGeneratedY2 << endl;
	// logs << "Total predcitions calls : " << numPredicitionCalls << endl;
	// logs << "Num overflows in data sampleCreation : " << numOverflowsInDataSampleCreation << endl;
	// logs.close();
	
	return 1;

}

void totalSupply()
{

	unsigned long long x, y, z;
	for (z = 1; z <= gv.numLayers; z++)
	{
		for (x = 1; x <= gv.rowEndIdx; x++)
		{
			unsigned partitionNumber = whichPartition(x);
			for (y = 1; y <= gv.colEndIdx; y++)
			{
				std::string coordinates = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
				unsigned defaultSupply = gv.dd.layerList[z - 1]->default_supply;
				int nonDefaultSupply = 0;
				unordered_map<string, NonDefaultSupply *>::iterator itr =
					gv.dd.ndsMap.find(coordinates);
				//logs << "ndsSupply" << gv.dd.ndsMap.size() << endl;
				if (itr != gv.dd.ndsMap.end())
				{

					nonDefaultSupply = gv.dd.ndsMap[coordinates]->value;
					//logs << "ndsSupply" << " "<<(gv.dd.ndsMap[coordinates]->value)<<endl;
				}
				unsigned totSupply = defaultSupply + nonDefaultSupply;
				gv.dd.gGrid_supply[z][x][y] = totSupply;
				
				// if(partitionNumber==1){
				// 	gv.dd.gGrid_supply_firstPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==2){
				// 	gv.dd.gGrid_supply_secondPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==3){
				// 	gv.dd.gGrid_supply_thirdPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==4){
				// 	gv.dd.gGrid_supply_fourthPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==5){
				// 	gv.dd.gGrid_supply_fifthPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==6){
				// 	gv.dd.gGrid_supply_sixthPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==7){
				// 	gv.dd.gGrid_supply_seventhPartition[z][x][y] = totSupply;
				// }else if(partitionNumber==8){
				// 	gv.dd.gGrid_supply_eighthPartition[z][x][y] = totSupply;
				// }
			}
		}
	}
}

unsigned netWirelengthDemand(Net *net, bool cal_demand, bool deleteSegment)
{
	//Returns the total wirelength of the design
	//Updates the demand values of grids, as per requirement from the net segments

	unordered_map<string, unsigned> netCovered; //can use the ZERO index values of the 3D array instead
	unordered_map<string, unsigned>::iterator netItr;
	list<Route *>::iterator j = ((net)->segmentList).begin();
	unsigned long long start_gp = 0, end_gp = 0;
	//start grid point and end grid point (common variable for any direction traversal
	unsigned long long x = 0, y = 0, z = 0;
	unsigned wirelength = 0;

	j = ((net)->segmentList).begin();

	if (j == ((net)->segmentList).end()) //No segments present then all the pins bound to be in the same grid
	{
		vector<Pin *>::iterator p = ((net)->pinList).begin();
		//Traversal of pinList not needed as pins in the same grid only, thus need an increase of demand in
		//the grid cell by 1 only
		z = (*p)->masterPin->layer;
		x = (*p)->inst->row;
		y = (*p)->inst->col;
		//logs<<"No Segment Demand "<< x <<" "<< y <<" "<<z <<" "<<endl;
		m.lock();
		gv.dd.gGrid_demand[z][x][y] +=
			cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		m.unlock();

		// unsigned partitionNumber = whichPartition(x);
		// if(partitionNumber==1){
		// 	gv.dd.gGrid_demand_firstPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_firstPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==2){
		// 	gv.dd.gGrid_demand_secondPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_secondPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==3){
		// 	gv.dd.gGrid_demand_thirdPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_thirdPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==4){
		// 	gv.dd.gGrid_demand_fourthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fourthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==5){
		// 	gv.dd.gGrid_demand_fifthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fifthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==6){
		// 	gv.dd.gGrid_demand_sixthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_sixthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==7){
		// 	gv.dd.gGrid_demand_seventhPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_seventhPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }else if(partitionNumber==8){
		// 	gv.dd.gGrid_demand_eighthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_eighthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
		// }

		wirelength++;
	}
	else
	{
		for (; j != ((net)->segmentList).end(); ++j)
		{
			Direction axis = (*j)->axis;
			if (axis == ALONG_ROW)
			{
				if ((*j)->sRow < (*j)->eRow)
				{
					start_gp = (*j)->sRow;
					end_gp = (*j)->eRow;
				}
				else
				{
					start_gp = (*j)->eRow;
					end_gp = (*j)->sRow;
				}

				y = (*j)->eCol;
				z = (*j)->eLayer;
				for (x = start_gp; x <= end_gp; x++)
				{
					string currentCoordinate = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
					netItr = netCovered.find(currentCoordinate);
					if (netItr == (netCovered).end())
					{
						m.lock();
						gv.dd.gGrid_demand[z][x][y] +=
							cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						m.unlock();

						// unsigned partitionNumber = whichPartition(x);
						// if(partitionNumber==1){
						// 	gv.dd.gGrid_demand_firstPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_firstPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==2){
						// 	gv.dd.gGrid_demand_secondPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_secondPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==3){
						// 	gv.dd.gGrid_demand_thirdPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_thirdPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==4){
						// 	gv.dd.gGrid_demand_fourthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fourthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==5){
						// 	gv.dd.gGrid_demand_fifthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fifthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==6){
						// 	gv.dd.gGrid_demand_sixthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_sixthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==7){
						// 	gv.dd.gGrid_demand_seventhPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_seventhPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==8){
						// 	gv.dd.gGrid_demand_eighthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_eighthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }

						wirelength++;
						netCovered[currentCoordinate] = 1;
					}
				}
			}
			else if (axis == ALONG_COL)
			{
				if ((*j)->sCol < (*j)->eCol)
				{
					start_gp = (*j)->sCol;
					end_gp = (*j)->eCol;
				}
				else
				{
					start_gp = (*j)->eCol;
					end_gp = (*j)->sCol;
				}
				x = (*j)->eRow;

				z = (*j)->eLayer;
				for (y = start_gp; y <= end_gp; y++)
				{
					string currentCoordinate = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
					netItr = netCovered.find(currentCoordinate);
					if (netItr == (netCovered).end())
					{
						m.lock();
						gv.dd.gGrid_demand[z][x][y] +=
							cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						m.unlock();

						// unsigned partitionNumber = whichPartition(x);
						// if(partitionNumber==1){
						// 	gv.dd.gGrid_demand_firstPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_firstPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==2){
						// 	gv.dd.gGrid_demand_secondPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_secondPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==3){
						// 	gv.dd.gGrid_demand_thirdPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_thirdPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==4){
						// 	gv.dd.gGrid_demand_fourthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fourthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==5){
						// 	gv.dd.gGrid_demand_fifthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fifthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==6){
						// 	gv.dd.gGrid_demand_sixthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_sixthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==7){
						// 	gv.dd.gGrid_demand_seventhPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_seventhPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==8){
						// 	gv.dd.gGrid_demand_eighthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_eighthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }

						wirelength++;
						netCovered[currentCoordinate] = 1;
					}
				}
			}
			else if (axis == ALONG_Z)
			{
				if ((*j)->sLayer < (*j)->eLayer)
				{
					start_gp = (*j)->sLayer;
					end_gp = (*j)->eLayer;
				}
				else
				{
					start_gp = (*j)->eLayer;
					end_gp = (*j)->sLayer;
				}
				x = (*j)->eRow;
				y = (*j)->eCol;

				for (z = start_gp; z <= end_gp; z++)
				{
					string currentCoordinate = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
					netItr = netCovered.find(currentCoordinate);
					if (netItr == (netCovered).end())
					{
						m.lock();
						gv.dd.gGrid_demand[z][x][y] +=
							cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						m.unlock();

						// unsigned partitionNumber = whichPartition(x);
						// if(partitionNumber==1){
						// 	gv.dd.gGrid_demand_firstPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_firstPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==2){
						// 	gv.dd.gGrid_demand_secondPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_secondPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==3){
						// 	gv.dd.gGrid_demand_thirdPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_thirdPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==4){
						// 	gv.dd.gGrid_demand_fourthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fourthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==5){
						// 	gv.dd.gGrid_demand_fifthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_fifthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==6){
						// 	gv.dd.gGrid_demand_sixthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_sixthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==7){
						// 	gv.dd.gGrid_demand_seventhPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_seventhPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }else if(partitionNumber==8){
						// 	gv.dd.gGrid_demand_eighthPartition[z][x][y] += cal_demand ? (deleteSegment ? ((gv.dd.gGrid_demand_eighthPartition[z][x][y] != 0) ? (-1) : 0) : (+1)) : 0;
						// }

						wirelength++;
						netCovered[currentCoordinate] = 1;
					}
				}
			}
		}
	}

	return wirelength;
}

unsigned totalNetWirelengthDemand()
{

	ofstream file;

	unsigned total_wirelength = 0;
	unsigned wirelength = 0;

	file.open("Routed_Wirelength_Demand.txt");

	for (vector<Net *>::iterator i = gv.dd.netList.begin();
		 i != gv.dd.netList.end(); ++i)
	{
		wirelength = 0;
		wirelength = netWirelengthDemand(*i, true, false);
		file << (*i)->name << ", gGrid length: " << wirelength << endl;
		total_wirelength += wirelength;
	}
	file << "Total gGrid length: " << total_wirelength << endl;
	file.close();

	return total_wirelength;
}

void getBlockDemand()
{
	for (vector<Inst *>::iterator i = gv.dd.instList.begin();
		 i != gv.dd.instList.end(); ++i)
	{
		for (vector<Blockage>::iterator j = (*i)->master->blockageList.begin();
			 j != (*i)->master->blockageList.end(); ++j)
		{
			m.lock();
			gv.dd.gGrid_demand[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			m.unlock();

			// unsigned partitionNumber = whichPartition((*i)->row);
			// if(partitionNumber==1){
			// 	gv.dd.gGrid_demand_firstPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==2){
			// 	gv.dd.gGrid_demand_secondPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==3){
			// 	gv.dd.gGrid_demand_thirdPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==4){
			// 	gv.dd.gGrid_demand_fourthPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==5){
			// 	gv.dd.gGrid_demand_fifthPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==6){
			// 	gv.dd.gGrid_demand_sixthPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==7){
			// 	gv.dd.gGrid_demand_seventhPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }else if(partitionNumber==8){
			// 	gv.dd.gGrid_demand_eighthPartition[(*j).layer][(*i)->row][(*i)->col] += (*j).demand;
			// }

		}
	}
}

unsigned getNoCellInGridCount(MasterCell *cellName,
							  vector<MasterCell *> cellList)
{
	unsigned count = 0;
	for (vector<MasterCell *>::iterator i = cellList.begin();
		 i != cellList.end(); i++)
	{
		if ((*i) == cellName)
		{
			count++;
		}
	}
	return count;
}

unsigned getNoCellInGridCount(MasterCell *cellName1, MasterCell *cellName2,
							  vector<MasterCell *> cellList, unsigned &count1, unsigned &count2)
{
	for (vector<MasterCell *>::iterator i = cellList.begin();
		 i != cellList.end(); i++)
	{
		if ((*i) == cellName1)
		{
			count1++;
		}
		else if ((*i) == cellName2)
		{
			count2++;
		}
	}
	return count1;
}
void getCellExtraDemand()
{

	unsigned sameGridDemand = 0;
	unsigned adjGridDemand = 0;

	unsigned x = 1, y = 1;

	//Note that cellName variables actually store only the pointer and not the name of the cell
	MasterCell *cellName1;
	MasterCell *cellName2;

	unsigned cell1 = 0, cell2 = 0, cellPairs = 0; //counter for cells present in a list
	unsigned cellPairPre = 0, cellPairNext = 0;
	unsigned cell1Current = 0, cell2Current = 0;
	unsigned cell1Previous = 0, cell2Previous = 0;
	unsigned cell1Next = 0, cell2Next = 0;

	vector<MasterCell *> cellsInCurrentGrid;
	vector<MasterCell *> cellsInNextGrid;
	vector<MasterCell *> cellsInPrevGrid;

	//	unordered_map<string, int> netCovered;
	for (x = 1; x <= gv.rowEndIdx; x++)
	{
		for (y = 1; y <= gv.colEndIdx; y++)
		{

			cellsInCurrentGrid.clear();
			cellsInPrevGrid.clear();
			cellsInNextGrid.clear();

			for (vector<Inst *>::iterator i = gv.dd.instList.begin();
				 i != gv.dd.instList.end(); ++i)
			{

				if ((*i)->row == x && (*i)->col == y)
				{ //can use nested if to check y==col
					cellsInCurrentGrid.push_back((*i)->master);
				}
				else if ((*i)->row == x && (*i)->col == y - 1)
				{
					cellsInPrevGrid.push_back((*i)->master);
				}
				else if ((*i)->row == x && (*i)->col == y + 1)
				{
					cellsInNextGrid.push_back((*i)->master);
				}
			}

			for (vector<ExtraDemand *>::iterator i =
					 gv.dd.extraDemandList.begin();
				 i != gv.dd.extraDemandList.end(); ++i)
			{

				cell1 = 0;
				cell2 = 0;
				cellPairPre = 0;
				cellPairNext = 0;
				cell1Current = 0;
				cell2Current = 0;
				cell1Previous = 0;
				cell2Previous = 0;
				cell1Next = 0;
				cell2Next = 0;

				cellName1 = (*i)->cell1;
				cellName2 = (*i)->cell2;

				if ((*i)->same == true)
				{

					getNoCellInGridCount(cellName1, cellName2,
										 cellsInCurrentGrid, cell1, cell2);
					//					int cell2 = getNoCellInGridCount(cellName2,
					//							cellsInCurrentGrid);
					cellPairs = (cell1 > cell2) ? cell2 : cell1;

					sameGridDemand = (cellPairs * (*i)->demand);
					m.lock();
					gv.dd.gGrid_demand[(*i)->layer->number][x][y] +=
						sameGridDemand;
					m.unlock();

					// unsigned partitionNumber = whichPartition(x);
					// if(partitionNumber==1){
					// 	gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==2){
					// 	gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==3){
					// 	gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==4){
					// 	gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==5){
					// 	gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==6){
					// 	gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==7){
					// 	gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }else if(partitionNumber==8){
					// 	gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] += sameGridDemand;
					// }

					sameGridDemand = 0;
				}

				else if ((*i)->same == false)
				{

					if (cellName1 == cellName2)
					{

						cell1Current = getNoCellInGridCount(cellName1,
															cellsInCurrentGrid);
						cell1Previous = getNoCellInGridCount(cellName1,
															 cellsInPrevGrid);
						cell1Next = getNoCellInGridCount(cellName1,
														 cellsInNextGrid);

						cellPairPre =
							(cell1Current > cell1Previous) ? cell1Previous : cell1Current;
						cellPairNext =
							(cell1Current > cell1Next) ? cell1Next : cell1Current;
					}
					else
					{
						getNoCellInGridCount(cellName1, cellName2,
											 cellsInCurrentGrid, cell1Current, cell2Current);
						getNoCellInGridCount(cellName1, cellName2,
											 cellsInPrevGrid, cell1Previous, cell2Previous);
						getNoCellInGridCount(cellName1, cellName2,
											 cellsInNextGrid, cell1Next, cell2Next);

						/////////////for finding cell pairs /////////////
						unsigned a =
							(cell1Current > cell2Previous) ? cell2Previous : cell1Current;
						unsigned b =
							(cell2Current > cell1Previous) ? cell1Previous : cell2Current;
						unsigned c =
							(cell1Current > cell2Next) ? cell2Next : cell1Current;
						unsigned d =
							(cell2Current > cell1Next) ? cell1Next : cell2Current;

						cellPairPre = a + b;
						cellPairNext = c + d;
					}
					adjGridDemand += (cellPairPre + cellPairNext) * (*i)->demand;
					m.lock();
					gv.dd.gGrid_demand[(*i)->layer->number][x][y] +=
						adjGridDemand;
					m.unlock();
					// unsigned partitionNumber = whichPartition(x);
					// if(partitionNumber==1){
					// 	gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==2){
					// 	gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==3){
					// 	gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==4){
					// 	gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==5){
					// 	gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==6){
					// 	gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==7){
					// 	gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }else if(partitionNumber==8){
					// 	gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] += adjGridDemand;
					// }
					adjGridDemand = 0;
				}
			}
		}
	}
}

void getDemand()
{
	std::memset(gv.dd.gGrid_demand, 0, sizeof(gv.dd.gGrid_demand)); // gGrig_demand is array for complete grid
	// std::memset(gv.dd.gGrid_demand_secondPartition, 0, sizeof(gv.dd.gGrid_demand_secondPartition)); 
	// std::memset(gv.dd.gGrid_demand_thirdPartition, 0, sizeof(gv.dd.gGrid_demand_thirdPartition)); 
	// std::memset(gv.dd.gGrid_demand_fourthPartition, 0, sizeof(gv.dd.gGrid_demand_fourthPartition)); 
	// std::memset(gv.dd.gGrid_demand_fifthPartition, 0, sizeof(gv.dd.gGrid_demand_fifthPartition)); // gGrig_demand is array for complete grid
	// std::memset(gv.dd.gGrid_demand_sixthPartition, 0, sizeof(gv.dd.gGrid_demand_sixthPartition)); 
	// std::memset(gv.dd.gGrid_demand_seventhPartition, 0, sizeof(gv.dd.gGrid_demand_seventhPartition)); 
	// std::memset(gv.dd.gGrid_demand_eighthPartition, 0, sizeof(gv.dd.gGrid_demand_eighthPartition)); 

	std::thread thread1(totalNetWirelengthDemand);
	std::thread thread2(getBlockDemand);
	std::thread thread3(getCellExtraDemand);
	thread1.join();
	thread2.join();
	thread3.join();
}

float getCongestion(unsigned x, unsigned y, unsigned z)
{

	unsigned totSupply = 0, totDemand = 0;
	float congestion = 0;
	
	totDemand = gv.dd.gGrid_demand[z][x][y];
	// unsigned partitionNumber = whichPartition(x);
	// if(partitionNumber==1){
	// 	totDemand = gv.dd.gGrid_demand_firstPartition[z][x][y];
	// }else if(partitionNumber==2){
	// 	totDemand = gv.dd.gGrid_demand_secondPartition[z][x][y];
	// }else if(partitionNumber==3){
	// 	totDemand = gv.dd.gGrid_demand_thirdPartition[z][x][y];
	// }else if(partitionNumber==4){
	// 	totDemand = gv.dd.gGrid_demand_fourthPartition[z][x][y];
	// }else if(partitionNumber==5){
	// 	totDemand = gv.dd.gGrid_demand_fifthPartition[z][x][y];
	// }else if(partitionNumber==6){
	// 	totDemand = gv.dd.gGrid_demand_sixthPartition[z][x][y];
	// }else if(partitionNumber==7){
	// 	totDemand = gv.dd.gGrid_demand_seventhPartition[z][x][y];
	// }else if(partitionNumber==8){
	// 	totDemand = gv.dd.gGrid_demand_eighthPartition[z][x][y];
	// }

	totSupply = gv.dd.gGrid_supply[z][x][y];

	// if(partitionNumber==1){
	// 	totSupply = gv.dd.gGrid_supply_firstPartition[z][x][y];
	// }else if(partitionNumber==2){
	// 	totSupply = gv.dd.gGrid_supply_secondPartition[z][x][y];
	// }else if(partitionNumber==3){
	// 	totSupply = gv.dd.gGrid_supply_thirdPartition[z][x][y];
	// }else if(partitionNumber==4){
	// 	totSupply = gv.dd.gGrid_supply_fourthPartition[z][x][y];
	// }else if(partitionNumber==5){
	// 	totSupply = gv.dd.gGrid_supply_fifthPartition[z][x][y];
	// }else if(partitionNumber==6){
	// 	totSupply = gv.dd.gGrid_supply_sixthPartition[z][x][y];
	// }else if(partitionNumber==7){
	// 	totSupply = gv.dd.gGrid_supply_seventhPartition[z][x][y];
	// }else if(partitionNumber==8){
	// 	totSupply = gv.dd.gGrid_supply_eighthPartition[z][x][y];
	// }


	congestion = (float)(((totDemand * 1.0) / totSupply) * 1.0);
	//<<"In getCongestion "<<x<<" " << y <<" "<<z<<" : "<< congestion << endl;
	return congestion;
}

int getOverFlow(unsigned x, unsigned y, unsigned z)
{

	unsigned totSupply = 0, totDemand = 0, overflow = 0;
	totSupply = gv.dd.gGrid_supply[z][x][y];
	totDemand = gv.dd.gGrid_demand[z][x][y];
	// unsigned partitionNumber = whichPartition(x);
	// if(partitionNumber==1){
	// 	totDemand = gv.dd.gGrid_demand_firstPartition[z][x][y];
	// }else if(partitionNumber==2){
	// 	totDemand = gv.dd.gGrid_demand_secondPartition[z][x][y];
	// }else if(partitionNumber==3){
	// 	totDemand = gv.dd.gGrid_demand_thirdPartition[z][x][y];
	// }else if(partitionNumber==4){
	// 	totDemand = gv.dd.gGrid_demand_fourthPartition[z][x][y];
	// }else if(partitionNumber==5){
	// 	totDemand = gv.dd.gGrid_demand_fifthPartition[z][x][y];
	// }else if(partitionNumber==6){
	// 	totDemand = gv.dd.gGrid_demand_sixthPartition[z][x][y];
	// }else if(partitionNumber==7){
	// 	totDemand = gv.dd.gGrid_demand_seventhPartition[z][x][y];
	// }else if(partitionNumber==8){
	// 	totDemand = gv.dd.gGrid_demand_eighthPartition[z][x][y];
	// }

	// if(partitionNumber==1){
	// 	totSupply = gv.dd.gGrid_supply_firstPartition[z][x][y];
	// }else if(partitionNumber==2){
	// 	totSupply = gv.dd.gGrid_supply_secondPartition[z][x][y];
	// }else if(partitionNumber==3){
	// 	totSupply = gv.dd.gGrid_supply_thirdPartition[z][x][y];
	// }else if(partitionNumber==4){
	// 	totSupply = gv.dd.gGrid_supply_fourthPartition[z][x][y];
	// }else if(partitionNumber==5){
	// 	totSupply = gv.dd.gGrid_supply_fifthPartition[z][x][y];
	// }else if(partitionNumber==6){
	// 	totSupply = gv.dd.gGrid_supply_sixthPartition[z][x][y];
	// }else if(partitionNumber==7){
	// 	totSupply = gv.dd.gGrid_supply_seventhPartition[z][x][y];
	// }else if(partitionNumber==8){
	// 	totSupply = gv.dd.gGrid_supply_eighthPartition[z][x][y];
	// }

	overflow =
		((int)(totDemand - totSupply) > 0) ? (int)(totDemand - totSupply) : 0;

	return overflow;
}

int parse(const string &ifname, ofstream &logs)
// How the data structures are getting formed is important
{
	string line;
	ifstream file(ifname.c_str()); //c_str returns a constant pointer to the string array of complete size
	unsigned cellId = 0;
	unsigned lno = 0;
	unsigned numDefaultG = 0;
	unsigned numMasterCell = 0;
	unsigned numMasterPinCount = 0, numMasterPin = 0;
	unsigned numBlockCount = 0;
	unordered_map<string, MasterPin *> mPinMap;
	vector<Blockage> blockList;
	vector<MasterPin *> masterPinList;
	string masterName;
	Net *currentNet = 0;
	unsigned numPins = 0;
	unsigned masterCellLayers = 0;
	unsigned pinIdGlobal;

	list<Inst *> netCellList;
	vector<Pin *> netPinList;

	bool parseRouting = false;

	bool firstPartition = false, secondPartition = false,
		thirdPartition = false, fourthPartition = false ,fifthPartition = false, sixthPartition = false, seventhPartition = false,
		 eighthPartition = false;
	string currNet;
	string prevNet;
	//	bool routeListPresent = false;
	if (file.is_open())
	{
		while (getline(file, line))
		{
			lno++; // ? With each entry into the loop we are getting one line
			stringstream ss(line);
			string tok;
			while (getline(ss, tok, ' '))
			{

				if (tok == "MaxCellMove")
				{
					getline(ss, tok, ' ');
					gv.maxCellMove = atoi(tok.c_str()); //atoi used to convert the text to integer value to be stored in the data
				}
				if (tok == "GGridBoundaryIdx")
				{
					getline(ss, tok, ' ');
					gv.rowBeginIdx = atoi(tok.c_str());

					getline(ss, tok, ' ');
					gv.colBeginIdx = atoi(tok.c_str());

					getline(ss, tok, ' ');
					gv.rowEndIdx = atoi(tok.c_str());

					getline(ss, tok, ' ');
					gv.colEndIdx = atoi(tok.c_str());

					if (gv.rowEndIdx > 15) // * if the size is above 50 then we will partition design into 8 parts
					// ! DOUBT: Why 8 parts?
					{
						firstPartitionStart = gv.rowBeginIdx;
						firstPartitionEnd = (gv.rowEndIdx / 2);
						secondPartitionStart = firstPartitionEnd + 1;
						secondPartitionEnd =( gv.rowEndIdx) ;
						/*thirdPartitionStart = secondPartitionEnd + 1;
						thirdPartitionEnd = (gv.rowEndIdx / 4) * 3;
						fourthPartitionStart = thirdPartitionEnd + 1;
						fourthPartitionEnd = (gv.rowEndIdx );*/
						/*fifthPartitionStart = fourthPartitionEnd + 1;
						fifthPartitionEnd = (gv.rowEndIdx / 8) * 5;
						sixthPartitionStart = fifthPartitionEnd + 1;
						sixthPartitionEnd = (gv.rowEndIdx / 8) * 6;
						seventhPartitionStart = sixthPartitionEnd + 1;
						seventhPartitionEnd = (gv.rowEndIdx / 8) * 7;
						eighthPartitionStart = seventhPartitionEnd + 1;
						eighthPartitionEnd = gv.rowEndIdx;*/
					}
					cout << "1st partition starts at :" << firstPartitionStart
						 << " and ends at :" << firstPartitionEnd << endl;
					cout << "2nd partition starts at :" << secondPartitionStart
						 << " and ends at :" << secondPartitionEnd << endl;
					/*cout << "3rd partition starts at :" << thirdPartitionStart
						 << " and ends at :" << thirdPartitionEnd << endl;
					cout << "4th partition starts at :" << fourthPartitionStart
						 << " and ends at :" << fourthPartitionEnd << endl;*/
					/*cout << "5th partition starts at :" << fifthPartitionStart
						 << " and ends at :" << fifthPartitionEnd << endl;
					cout << "6th partition starts at :" << sixthPartitionStart
						 << " and ends at :" << sixthPartitionEnd << endl;
					cout << "7th partition starts at :" << seventhPartitionStart
						 << " and ends at :" << seventhPartitionEnd << endl;
					cout << "8th partition starts at :" << eighthPartitionStart
						 << " and ends at :" << eighthPartitionEnd << endl;*/
				}
				if (tok == "NumLayer")
				{
					getline(ss, tok, ' ');
					gv.numLayers = atoi(tok.c_str());
					gv.l = gv.numLayers;
				}

				if (tok == "Lay")
				{
					getline(ss, tok, ' ');
					string layName = tok;

					getline(ss, tok, ' ');
					unsigned layNo = atoi(tok.c_str());

					getline(ss, tok, ' ');
					bool horizontal = (tok == "H") ? true : false;

					getline(ss, tok, ' ');
					unsigned defaultS = atoi(tok.c_str());

					Layer *l = new Layer(layName, layNo, horizontal, defaultS);

					gv.dd.layerList.push_back(l);
					gv.dd.layerMap[layName] = l;
				}

				if (tok == "NumNonDefaultSupplyGGrid")
				{
					getline(ss, tok, ' ');
					numDefaultG = gv.numNonDefaultSupplyGGrid = atoi(tok.c_str());
					break;
				}
				//why "nested if" not used??, as traditionally taking every new line from file using while loop only

				if (numDefaultG > 0)
				{
					unsigned row = atoi(tok.c_str());

					getline(ss, tok, ' '); //to get the data in the same line of text,separated by ' ' space
					unsigned column = atoi(tok.c_str());

					getline(ss, tok, ' '); //to get the data in the same line of text,separated by ' ' space
					unsigned layer = atoi(tok.c_str());

					getline(ss, tok, ' '); //to get the data in the same line of text,separated by ' ' space
					int value = atoi(tok.c_str());

					NonDefaultSupply *nds = new NonDefaultSupply(row, column,
																 layer, value);

					unsigned long long x, y, z;
					x = row;
					y = column;
					z = layer;
					string coordinate = std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);

					unordered_map<string, NonDefaultSupply *>::iterator net; // ? We want to locate the nonDefSup of cell based on coordinates
					net = gv.dd.ndsMap.find(coordinate);
					if (net == gv.dd.ndsMap.end())
					{
						gv.dd.ndsMap[coordinate] = nds; // Add if not present in the map
					}
					else
					{
						net->second->value += nds->value; // Otherwise update the value
					}

					// ! DOUBT

					gv.dd.ndsList.push_back(nds);
					--numDefaultG;
				}

				if (tok == "NumMasterCell")
				{
					getline(ss, tok, ' ');
					numMasterCell = gv.masterCellCount = atoi(tok.c_str());
					break;
				}

				if (numMasterCell > 0)
				{
					if (tok == "MasterCell")
					{
						masterPinList.clear();
						masterCellLayers = 0;
						getline(ss, tok, ' ');
						masterName = tok;

						getline(ss, tok, ' ');
						numMasterPin = numMasterPinCount = atoi(tok.c_str());
						cout << "\nnumMasterPin:" << masterName << " -->" << numMasterPin << endl;

						getline(ss, tok, ' ');
						numBlockCount = atoi(tok.c_str());

						mPinMap.clear();
						blockList.clear();

						break;
					}

					if (tok == "Pin")
					{

						getline(ss, tok, ' ');
						string name = tok;

						getline(ss, tok, ' ');
						unsigned layer = atoi(tok.c_str() + 1); // Skip M from M1, M2 etc...
						masterCellLayers |= (0x01 << (layer - 1));
						unsigned masterPinIndex = numMasterPin - numMasterPinCount + 1;
						MasterPin *mp = new MasterPin(masterPinIndex, name, layer);
						mPinMap[name] = mp;
						cout << "Pin " << name << " included in the masterPinMap" << endl;
						masterPinList.push_back(mp);

						numMasterPinCount--;
					}

					if (tok == "Blkg")
					{

						getline(ss, tok, ' ');
						string name = tok;

						getline(ss, tok, ' ');
						unsigned layer = atoi(tok.c_str() + 1); // Skip M from M1, M2 etc...
						masterCellLayers |= (0x01 << (layer - 1));

						getline(ss, tok, ' ');
						unsigned demand = atoi(tok.c_str()); // Skip M from M1, M2 etc...
						Blockage b(name, layer, demand);
						blockList.push_back(b);

						numBlockCount--;
					}

					if (numMasterPinCount == 0 && numBlockCount == 0)
					{
						MasterCell *mc = new MasterCell(masterName, masterCellLayers, numMasterPin, mPinMap, blockList, masterPinList);
						gv.dd.masterCellList.push_back(mc);
						gv.dd.mcMap[masterName] = mc;
						numMasterCell--;
					}
				}

				if (tok == "NumNeighborCellExtraDemand")
				{
					getline(ss, tok, ' ');
					gv.numNeighborCellExtraDemand = atoi(tok.c_str());
					break;
				}

				if (tok == "sameGGrid" || tok == "adjHGGrid")
				{
					bool same = (tok == "sameGGrid");
					getline(ss, tok, ' ');
					MasterCell *mc1 = gv.dd.getMasterCellFromName(tok);
					getline(ss, tok, ' ');
					MasterCell *mc2 = gv.dd.getMasterCellFromName(tok);
					getline(ss, tok, ' ');
					Layer *l = gv.dd.getLayerFromName(tok);
					getline(ss, tok, ' ');
					int demand = atoi(tok.c_str());
					ExtraDemand *ed = new ExtraDemand(mc1, mc2, l, demand,
													  same);
					gv.dd.extraDemandList.push_back(ed);
				}

				if (tok == "NumCellInst")
				{
					getline(ss, tok, ' ');
					gv.numCellInst = atoi(tok.c_str());
					break;
				}

				if (tok == "CellInst")
				{
					getline(ss, tok, ' ');
					string name = tok;
					getline(ss, tok, ' ');
					MasterCell *mc = gv.dd.getMasterCellFromName(tok);
					// create a pinlist here
					//
					/*
					cout << "\n\n\nI am here 1\n\n\n";
					unordered_map<string, Pin *> pinMap;
					cout << "\n\n\nI am here 2\n\n\n";
					for (vector<MasterPin *>::iterator i = mc->masterPinList.begin(); i != mc->masterPinList.end(); i++)
					{

						assert(*i);

						string pinName = (*i)->name;
						MasterPin* currentMasterPin = (*i);
						Pin* newPin = Pin(pinName, currentMasterPin);
						cout << (*i)->name << endl;
						//newPin->name= (*i)->name;
						cout << "\n\n\nI am here 3\n\n\n";
						//newPin->inst = NULL;
						//p->name = pinName;

						//newPin->masterPin = (*i);
						pinMap[(*i)->name] = newPin;
						//pinMap[pinName] = p;
					}
					cout << "\n\n\nI am here 4\n\n\n";*/
					getline(ss, tok, ' ');
					unsigned row = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned col = atoi(tok.c_str());
					getline(ss, tok, ' ');
					bool movable = (tok == "Movable");
					cellId++;
					//cout << "\n\n\nI am here 5\n\n\n";
					Inst *inst = new Inst(cellId, name, mc, row, col, movable);
					gv.dd.instList.push_back(inst);
					inst->inFixedPartition = false;
					inst->cellIndex = 0; //ML

					if (gv.rowEndIdx > 15)
					{
						if (row <= firstPartitionEnd)
						{
							gv.dd.instListFirstPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 1"<<endl;
						}
						else if (row <= secondPartitionEnd)
						{
							gv.dd.instListSecondPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 2"<<endl;
						}
						/*else if (row <= thirdPartitionEnd)
						{
							gv.dd.instListThirdPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 3"<<endl;
						}
						else if (row <= fourthPartitionEnd)
						{
							gv.dd.instListFourthPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 4"<<endl;
						}*/
						/*else if (row <= fifthPartitionEnd)
						{
							gv.dd.instListFifthPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 5"<<endl;
						}
						else if (row <= sixthPartitionEnd)
						{
							gv.dd.instListSixthPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 6"<<endl;
						}
						else if (row <= seventhPartitionEnd)
						{
							gv.dd.instListSeventhPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 7"<<endl;
						}
						else if (row <= eighthPartitionEnd)
						{
							gv.dd.instListEighthPartition.push_back(inst);
							//<<"Cell "<<inst->name<<" pushed to partition 8"<<endl;
						}*/
					}

					gv.dd.instMap[name] = inst;
					break;
				}

				if (tok == "Net")
				{
					netCellList.clear();
					netPinList.clear();

					getline(ss, tok, ' ');
					string name = tok;
					getline(ss, tok, ' ');
					numPins = atoi(tok.c_str());
					getline(ss, tok, ' ');
					Layer *layer = gv.dd.getLayerFromName(tok);

					Net *net = new Net(name, numPins, layer);
					net->inFixedPartition = false;
					currentNet = net;
					gv.dd.netList.push_back(net);
					gv.dd.netMap[name] = net;
					break;
				}

				if (currentNet && (tok == "Pin"))
				{
					getline(ss, tok, ' ');
					string pinName = tok;
					//cout << endl
					// << "Trying to get the pin " << tok;
					Pin *pin = gv.dd.getPinFromName(tok); // Does not run for all the pins---> this is a bug
					pin->net = currentNet;
					stringstream ssInstName(tok);
					string instName;
					getline(ssInstName, instName, '/');
					Inst *inst = gv.dd.getInstanceFromName(instName);
					assert(inst);
					inst->netList.push_back(currentNet);
					vector<Pin *>::iterator netPinItr = netPinList.begin();
					for (; netPinItr != netPinList.end(); netPinItr++)
					{
						pin->pinAdjacencyList.push_back(*netPinItr);   // Put all previous pins in adjacency of new pin
						(*netPinItr)->pinAdjacencyList.push_back(pin); // Put new pin in adjacency of pins already connected to the net

						(*netPinItr)->inst->adjacencyList.push_back(inst);
						inst->adjacencyList.push_back((*netPinItr)->inst);
					}
					netPinList.push_back(pin);

					

					netCellList.push_back(inst);
					currentNet->pinList.push_back(pin);
					break;
				}

				if (tok == "NumRoutes")
				{
					getline(ss, tok, ' ');
					gv.numRoutes = atoi(tok.c_str());
					parseRouting = true;
					break;
				}

				if (parseRouting)
				{

					unsigned sRow = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned sCol = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned sLayer = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned eRow = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned eCol = atoi(tok.c_str());
					getline(ss, tok, ' ');
					unsigned eLayer = atoi(tok.c_str());
					getline(ss, tok, ' ');

					Net *net = gv.dd.netMap[tok];
					assert(net);
					Direction axis;
					if (sRow != eRow)
					{
						axis = ALONG_ROW;
					}
					else if (sCol != eCol)
					{
						axis = ALONG_COL;
					}
					else if (sLayer != eLayer)
					{
						axis = ALONG_Z;
					}
					else
					{
						axis = UNDEF;
					}

					Route *r = new Route(sRow, sCol, sLayer, eRow, eCol, eLayer,
										 axis, net);

					gv.dd.routeList.push_back(r);

					net->segmentList.push_back(r);
					currNet = net->name;
					/*if (prevNet == "N697") {
						logs << "N697" << endl;
					}*/

					// ! DOUBT: What does this loop do

					if (currNet != "" && prevNet != "" && currNet != prevNet)
					{
						if (firstPartition && !secondPartition/* && !thirdPartition && !fourthPartition*/ /*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
						{
							// << prevNet << " belongs to p1" << endl;
							gv.dd.firstPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							/*thirdPartition = false;
							fourthPartition = false;*/
							/*fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;*/
						}
						else if (!firstPartition && secondPartition/* && !thirdPartition && !fourthPartition *//*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
						{
							// << prevNet << " belongs to p2" << endl;
							gv.dd.secondPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							/*thirdPartition = false;
							fourthPartition = false;*/
							/*fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;*/
						}
						// else if (!firstPartition && !secondPartition/* && thirdPartition && !fourthPartition *//*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
						// {
						// 	// << prevNet << " belongs to p3" << endl;
						// 	gv.dd.thirdPartitionNetList.push_back(
						// 		gv.dd.getNetFromName(prevNet));

						// 	firstPartition = false;
						// 	secondPartition = false;
						// 	thirdPartition = false;
						// 	fourthPartition = false;
						// 	/*fifthPartition = false;
						// 	sixthPartition = false;
						// 	seventhPartition = false;
						// 	eighthPartition = false;*/
						// }
						// else if (!firstPartition && !secondPartition && !thirdPartition && fourthPartition /*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
						// {
						// 	// << prevNet << " belongs to p4" << endl;
						// 	gv.dd.fourthPartitionNetList.push_back(
						// 		gv.dd.getNetFromName(prevNet));

						// 	firstPartition = false;
						// 	secondPartition = false;
						// 	thirdPartition = false;
						// 	fourthPartition = false;
						// 	/*fifthPartition = false;
						// 	sixthPartition = false;
						// 	seventhPartition = false;
						// 	eighthPartition = false;*/
						// }
						/*else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition)
						{
							// << prevNet << " belongs to p5" << endl;
							gv.dd.fifthPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							thirdPartition = false;
							fourthPartition = false;
							fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;
						}
						else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && sixthPartition && !seventhPartition && !eighthPartition)
						{
							// << prevNet << " belongs to p6" << endl;
							gv.dd.sixthPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							thirdPartition = false;
							fourthPartition = false;
							fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;
						}
						else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && !sixthPartition && seventhPartition && !eighthPartition)
						{
							// << prevNet << " belongs to p7" << endl;
							gv.dd.seventhPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							thirdPartition = false;
							fourthPartition = false;
							fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;
						}
						else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && !sixthPartition && !seventhPartition && eighthPartition)
						{
							// << prevNet << " belongs to p8" << endl;
							gv.dd.eighthPartitionNetList.push_back(
								gv.dd.getNetFromName(prevNet));

							firstPartition = false;
							secondPartition = false;
							thirdPartition = false;
							fourthPartition = false;
							fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;
						}*/
						else
						{
							// << prevNet << " is fixed net" << endl;
							Net *net = gv.dd.getNetFromName(prevNet);
							net->inFixedPartition = true;
							gv.dd.fixedNetList.push_back(net);
							firstPartition = false;
							secondPartition = false;
							// thirdPartition = false;
							// fourthPartition = false;
							/*fifthPartition = false;
							sixthPartition = false;
							seventhPartition = false;
							eighthPartition = false;*/
						}
					}
					// Same can also fall inside 2 partitions though
					if (((sRow >= firstPartitionStart && sRow <= firstPartitionEnd) || (eRow >= firstPartitionStart && eRow <= firstPartitionEnd)) && !firstPartition)
					{
						firstPartition = true;
						//currNet = net->name;
					}
					if (((sRow >= secondPartitionStart && sRow <= secondPartitionEnd) || (eRow >= secondPartitionStart && eRow <= secondPartitionEnd)) && !secondPartition)
					{
						secondPartition = true;
						//currNet = net->name;
					}

					// if (((sRow >= thirdPartitionStart && sRow <= thirdPartitionEnd) || (eRow >= thirdPartitionStart && eRow <= thirdPartitionEnd)) && !thirdPartition)
					// {
					// 	thirdPartition = true;
					// 	//currNet = net->name;
					// }
					// if (((sRow >= fourthPartitionStart && sRow <= fourthPartitionEnd) || (eRow >= fourthPartitionStart && eRow <= fourthPartitionEnd)) && !fourthPartition)
					// {
					// 	fourthPartition = true;
					// 	//currNet = net->name;
					// }
					/*if (((sRow >= fifthPartitionStart && sRow <= fifthPartitionEnd) || (eRow >= fifthPartitionStart && eRow <= fifthPartitionEnd)) && !fifthPartition)
					{
						fifthPartition = true;
						//currNet = net->name;
					}
					if (((sRow >= sixthPartitionStart && sRow <= sixthPartitionEnd) || (eRow >= sixthPartitionStart && eRow <= sixthPartitionEnd)) && !sixthPartition)
					{
						sixthPartition = true;
						//currNet = net->name;
					}
					if (((sRow >= seventhPartitionStart && sRow <= seventhPartitionEnd) || (eRow >= seventhPartitionStart && eRow <= seventhPartitionEnd)) && !seventhPartition)
					{
						seventhPartition = true;
						//currNet = net->name;
					}
					if (((sRow >= eighthPartitionStart && sRow <= eighthPartitionEnd) || (eRow >= eighthPartitionStart && eRow <= eighthPartitionEnd)) && !eighthPartition)
					{
						eighthPartition = true;
						//currNet = net->name;
					}*/

					prevNet = currNet; // * For the first time current and previous net are the same
				}
			}
		}
		
		file.close();
	}
	else
	{
		logs << "Unable to open input file: " << ifname << endl;
	}
	if (parseRouting)
	{
		if (firstPartition && !secondPartition /*&& !thirdPartition && !fourthPartition *//* && !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
		{
			// << prevNet << " belongs to p1" << endl;
			gv.dd.firstPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			// thirdPartition = false;
			// fourthPartition = false;
			/*fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;*/
		}
		else if (!firstPartition && secondPartition /*&&!thirdPartition && !fourthPartition */ /*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
		{
			// << prevNet << " belongs to p2" << endl;
			gv.dd.secondPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			// thirdPartition = false;
			// fourthPartition = false;
			/*fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;*/
		}
		else if (!firstPartition && !secondPartition /*&& thirdPartition && !fourthPartition *//*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
		{
			// << prevNet << " belongs to p3" << endl;
			gv.dd.thirdPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			// thirdPartition = false;
			// fourthPartition = false;
			/*fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;*/
		}
		else if (!firstPartition && !secondPartition /*&& !thirdPartition && fourthPartition*/ /*&& !fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition*/)
		{
			// << prevNet << " belongs to p4" << endl;
			gv.dd.fourthPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			// thirdPartition = false;
			// fourthPartition = false;
			/*fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;*/
		}
		/*else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && fifthPartition && !sixthPartition && !seventhPartition && !eighthPartition)
		{
			// << prevNet << " belongs to p5" << endl;
			gv.dd.fifthPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			thirdPartition = false;
			fourthPartition = false;
			fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;
		}
		else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && sixthPartition && !seventhPartition && !eighthPartition)
		{
			// << prevNet << " belongs to p6" << endl;
			gv.dd.sixthPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			thirdPartition = false;
			fourthPartition = false;
			fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;
		}
		else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && !sixthPartition && seventhPartition && !eighthPartition)
		{
			// << prevNet << " belongs to p7" << endl;
			gv.dd.seventhPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			thirdPartition = false;
			fourthPartition = false;
			fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;
		}
		else if (!firstPartition && !secondPartition && !thirdPartition && !fourthPartition && !fifthPartition && !sixthPartition && !seventhPartition && eighthPartition)
		{
			// << prevNet << " belongs to p8" << endl;
			gv.dd.eighthPartitionNetList.push_back(
				gv.dd.getNetFromName(prevNet));

			firstPartition = false;
			secondPartition = false;
			thirdPartition = false;
			fourthPartition = false;
			fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;
		}*/
		else
		{
			// << prevNet << " is fixed net" << endl;
			Net *net = gv.dd.getNetFromName(prevNet);
			net->inFixedPartition = true;
			gv.dd.fixedNetList.push_back(net);
			firstPartition = false;
			secondPartition = false;
			// thirdPartition = false;
			// fourthPartition = false;
			/*fifthPartition = false;
			sixthPartition = false;
			seventhPartition = false;
			eighthPartition = false;*/		}
		parseRouting = false;
	}
	logs << "Alloting all the pins their local pin Ids." << endl;
	unsigned pinIdCounter = 0;
	for (vector<Inst *>::iterator itr = gv.dd.instList.begin(); itr != gv.dd.instList.end(); itr++)
	{
		pinIdCounter = 0;
		for (unordered_map<string, Pin *>::iterator i = (*itr)->pinMap.begin(); i != (*itr)->pinMap.end(); i++)
		{
			pinIdCounter++;
			//(*i).second->pinIndexLocal = pinIdCounter;

			/*for (vector<Pin *>::iterator pItr = (*i).second->adjacencyList.begin(); pItr != (*i).second->adjacencyList.end(); pItr++)
			{
				cout << (*pItr)->inst->name << "/" << (*pItr)->masterPin->name << " ";
			}*/
		}
	}
	logs << lno << " lines in file " << ifname << " parsed." << endl;
	return 1;
}

int dumpOutput(ostream &os)
{
	os << "MaxCellMove " << gv.maxCellMove << endl;
	os << "GGridBoundaryIdx " << gv.rowBeginIdx << " " << gv.colBeginIdx << " "
	   << gv.rowEndIdx << " " << gv.colEndIdx << endl;
	os << "NumLayer " << gv.numLayers << endl;
	assert(gv.numLayers == gv.dd.layerList.size());

	gv.dd.write(os);
	return 1;
}

int generateDemandSuppVals()
{
	unsigned x, y, z;
	ofstream gridDmdSuppVals;
	// gridDemandVals.open("gridDemandValuesTabular.txt");
	gridDmdSuppVals.open("gridDemandandSupplyValues.txt");

	gridDmdSuppVals << "row col lay supply demand" << endl;
	// gridDemandVals << "Grid Demand Values Table\t(xyz):\t Value" << endl;

	// log << "row col lay supply demand" << endl;

	//getDemand();

	for (z = 1; z <= gv.numLayers; z++)
	{
		for (x = gv.rowBeginIdx; x <= gv.rowEndIdx; x++)
		{
			//unsigned partitionNumber = whichPartition(x);
			for (y = gv.colBeginIdx; y <= gv.colEndIdx; y++)
			{
				// if(partitionNumber==1){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " " << gv.dd.gGrid_supply_firstPartition[z][x][y] << " " << gv.dd.gGrid_demand_firstPartition[z][x][y] << endl;
				// }else if(partitionNumber==2){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_secondPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_secondPartition[z][x][y] << endl;
				// }else if(partitionNumber==3){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_thirdPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_thirdPartition[z][x][y] << endl;
				// }else if(partitionNumber==4){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_fourthPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_fourthPartition[z][x][y] << endl;
				// }else if(partitionNumber==5){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_fifthPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_fifthPartition[z][x][y] << endl;
				// }else if(partitionNumber==6){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_sixthPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_sixthPartition[z][x][y] << endl;
				// }else if(partitionNumber==7){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_seventhPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_seventhPartition[z][x][y] << endl;
				// }else if(partitionNumber==8){
				// 	gridDmdSuppVals << x << " " << y << " " << z << " "
				// 				<< gv.dd.gGrid_supply_eighthPartition[z][x][y] << " "
				// 				<< gv.dd.gGrid_demand_eighthPartition[z][x][y] << endl;
				// }
				gridDmdSuppVals << x << " " << y << " " << z << " "
								<< gv.dd.gGrid_supply[z][x][y] << " "
								<< gv.dd.gGrid_demand[z][x][y] << endl;
				// gridDemandVals << "\t(" << x << y << z << "):\t"
				// 			   << gv.dd.gGrid_demand[z][x][y];
			}
			//logs<<endl;
			//gridDemandVals << endl;
		}
		//logs<<endl;
		//gridDemandVals << endl;
	}
	//gridDemandVals.close();
	gridDmdSuppVals.close();

	return 1;
}

void getGridCongestion()
{
	float congestion;
	unsigned row, col, lay;
	for (lay = 1; lay <= gv.numLayers; lay++)
	{
		for (row = 1; row <= gv.rowEndIdx; row++)
		{
			for (col = 1; col <= gv.colEndIdx; col++)
			{
				congestion = getCongestion(row, col, lay);
				if (congestion >= CONGESTION_THRESHOLD)
				{
					gv.congestion_matrix[lay][row][col] = 1;
				}
				else
				{
					gv.congestion_matrix[lay][row][col] = 0;
				}
			}
		}
	}
}
void clearPinConnectivityMatrix(bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL])
{
	for (unsigned i = 1; i <= 100; i++)
	{
		for (unsigned j = 1; j <= 100; j++)
			/*gv.*/ pin_connectivity_matrix[i][j] = 0;
	}
}

unsigned findPinIndexForWindow(Pin *pin, unsigned *cellIdxList)
{
	//cout << "\n______________Inside findPinIndexForWindow()____________\n";
	//cout << "\nFinding pin index for pin : " << pin->inst->name << "/" << pin->name << ":-" << endl;
	Inst *inst = pin->inst;
	Inst *instItr;
	unsigned numPinsBeforeCurrentCell = 0;
	for (int i = 1; /*instItr != inst*/; i++)
	{
		instItr = gv.dd.getInstanceFromCellIndex(/*gv.*/ cellIdxList[i]);
		if (instItr == inst)
			break;

		//cout << instItr->master->numMasterPins << "(" << instItr->name << ")"
		//<< "+";
		numPinsBeforeCurrentCell += instItr->master->numMasterPins;
	}

	//cout << atoi((pin->name).c_str() + 1) << "=";
	unsigned pinIndex = numPinsBeforeCurrentCell + atoi((pin->name).c_str() + 1);
	//cout << pinIndex << endl;
	return pinIndex;
}
int createPinConnectivityForDesign(bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL], unsigned *cellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k)
{

	cout << "\n__________________PinConnectivity Creation____________________\n";
	/*
	We will be having the cellIdxList array ready. The forst index present in the array is basically the cell index 1 in the data that will be passed:-
	-> Access each cell corresponding to the cell index.
	-> Access the pin list --- This way, go to each and every pin in the current design.
	-> Then inside the pin, check all the adjacent pins-> instances and if the cell is present in cellIdx list then these two pins need to be connected.
	-> Now, you have two cells and their pins to be connected.

	? IMPORTANT POINT:-
	Here, you should not calculate the indexes in pin matrix using the cellIdx. Rather, use cellCount since that is the order in which data will be feeded to the
	design.

	For calculating the index of the pin connectivity matrix:-
	Traverse the cell index list until the current cell is reached and keep adding the subsequent numpins of the cells that are encountered. Finally find the
	index by adding the Pin name subscript using atoi.
	*
	 */
	pinCount = 0;
	int cCount = 1;
	if (/*gv.*/ k <= 1)
		cout << "\nLess than or equal to 1 cells in the design";

	for (; (cCount <= /*gv.*/ k) /*&& (gv.cellIdxList[cCount] != 0)*/; cCount++)
	{
		//cout << "\n\n**** AT CELLINDEX : " << cCount << "****" << endl
		//<< endl;

		pinCount += /*gv.*/ numPinsArray[cCount]; // Calculating pinCOunt

		unsigned currPinIdx, adjPinIdx;
		Inst *currentCell = gv.dd.getInstanceFromCellIndex(/*gv.*/ cellIdxList[cCount]);
		//assert(currentCell);
		//cout<< "\nCurrent cell name : " << currentCell->name<<endl;
		for (unordered_map<string, Pin *>::iterator i = currentCell->pinMap.begin(); i != currentCell->pinMap.end(); i++)
		{
			bool firstPin = true;
			//cout << "\nCurrent pin name : " << (*i).second->name << endl;
			for (vector<Pin *>::iterator itr = (*i).second->pinAdjacencyList.begin(); itr != (*i).second->pinAdjacencyList.end(); itr++)
			{
				//cout << "\nAdjacent pin name : " << (*itr)->inst->name << "/" << (*itr)->name << endl;
				bool adjCellPresentinCellIdxList = isCellPresentInCellIdxList((*itr)->inst->cellIndex, cellIdxList);
				if (adjCellPresentinCellIdxList)
				{
					//unsigned currPinIdx, adjPinIdx;

					currPinIdx = findPinIndexForWindow((*i).second, cellIdxList);
					if (firstPin)
					{
						/*gv.*/ pinIndexList[cCount].low = currPinIdx;
						/*gv.*/ pinIndexList[cCount].high = /*gv.*/ pinIndexList[cCount].low + currentCell->pinMap.size();

						//cout << "Found lowpinID = " << /*gv.*/pinIndexList[cCount].low << " and hipinIDX = " << /*gv.*/pinIndexList[cCount].high << " for cell : " << currentCell->name << endl;
						firstPin = false;
					}
					adjPinIdx = findPinIndexForWindow(*itr, cellIdxList);
					/*gv.*/ pin_connectivity_matrix[currPinIdx][adjPinIdx] = 1;
					/*gv.*/ pin_connectivity_matrix[adjPinIdx][currPinIdx] = 1;
				}
				//int max;
			}
		}
	}
	cout << "Pin connectivity matrix created :-" << endl;
	for (unsigned i = 1; i <= pinCount; i++)
	{
		for (unsigned j = 1; j <= pinCount; j++)
		{
			cout << pin_connectivity_matrix[i][j] << " ";
		}
		cout << "\n";
	}
	if (pinCount > maxPins)
		maxPins = pinCount;

	cout << "maxPins = " << maxPins << endl;
	//cout << "\nAt index :" << cCount;
	//pinCount = pinCount + /*gv.*/numPinsArray[cCount];
	//cout << "\nNext numpin value = " << /*gv.*/numPinsArray[cCount];
	if (pinCount > 56) // ! numPinsArray needs to be initialized
	{
		cCount--;
		cout << "\nCurrent numCells : " << /*gv.*/ k;
		cout << "\ncCount : " << cCount;
		cout << "\nFollowing is the status of numPinsArray";
		for (int i = 1; i <= NUM_CELLS; i++)
			cout << /*gv.*/ numPinsArray[i] << " ";
		cout << "\nPinCount = " << pinCount << "\nnumpinsArray value : " << /*gv.*/ numPinsArray[cCount];
		cout << endl
			 << "For cell " << gv.dd.getInstanceFromCellIndex(/*gv.*/ cellIdxList[cCount])->name;
		cout << "\nPincount > 56 while incrementing";

		cout << endl;
		cout << "Cell indices and pin indices:-\n";
		for (int i = 1; (i <= /*gv.*/ k) /*&& (gv.cellIdxList[cCount] != 0)*/; i++)
		{
			//cout << "Inside loop";
			Inst *theCell = gv.dd.getInstanceFromCellIndex(/*gv.*/ cellIdxList[i]);
			cout << "Cell : " << theCell->name << "(" << i << ")";
			cout << "\nPins : ";
			for (unordered_map<string, Pin *>::iterator itr = theCell->pinMap.begin(); itr != theCell->pinMap.end(); itr++)
			{
				cout << (*itr).first << "(" << (*itr).second->pinIndexLocal << ")"
					 << ",";
			}
			cout << endl;
		}
		return -1;
		////(2);
	}
	//else
	//cout << "\nCurrent pin count = " << pinCount;
	return 1;
}
void clearNumPinsArray()
{
	for (int i = 1; i <= NUM_CELLS; i++)
	{
		gv.numPinsArray[i] = 0;
	}
}
int partitionDesign(vector<Inst *> &instList, ofstream &dataFile, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex,
					unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
					list<Inst *> &cellsInWindow, list<Inst *> &cellsInRowWindow, list<Inst *> &cellsBeyondRowWindow)
{ // ? HEre e pass only the present window after sliding by a predetermined amount
	unsigned cellRow = 0, cellCol = 0;
	//unsigned numPinsArray[NUM_CELLS + 1];
	struct pinIndexes pinIndexList[NUM_CELLS + 1];
	unsigned cellIdxList[NUM_CELLS + 1];		// Done
	unsigned rowPosCell[NUM_CELLS + 1];			// Done
	unsigned colPosCell[NUM_CELLS + 1];			// Done
	unsigned demandValCell[NUM_CELLS + 1];		// Done
	unsigned movableCellIdxList[NUM_CELLS + 1]; // Done
	unsigned numPinsArray[NUM_CELLS + 1];		// Done
	bool connectivity_matrix[2001][2001];		// Done

	unsigned k = 0; // numCells
	list<Inst *> movableCellList;
	clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample

	list<Inst *>::iterator itr = /*gv.*/ cellsInRowWindow.begin();
	bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL];

	//vector <Inst *>::iterator itr;
	unsigned cellIdx = 0;
	cout << "Window: " << startGridRowIdx << " " << endGridRowIdx << " " << startGridColIdx << " " << endGridColIdx << endl;
	clearPinConnectivityMatrix(pin_connectivity_matrix);
	if (startGridRowIdx == /*gv.rowBeginIdx*/ rowStartIndex && startGridColIdx == /*gv.colBeginIdx*/ colStartIndex) // *  first time partition, starting at origin
	{
		for (vector<Inst *>::iterator instListItr = /*gv.dd.*/ instList.begin(); instListItr != /*gv.dd.*/ instList.end(); ++instListItr)
		{
			//cout<<"Inside for loop 1"<<endl;
			cellRow = (*instListItr)->row;
			cellCol = (*instListItr)->col;

			// cout<<"Cell: "<<(*instListItr)->name<<endl;
			if (cellRow >= startGridRowIdx && cellRow <= endGridRowIdx) // ?  The row window check
			{	
				/*gv.*/ cellsInRowWindow.push_back(*instListItr);
				//cout<<" is in row window "<<startGridRowIdx<<" "<<endGridRowIdx<<endl;

				if (cellCol >= startGridColIdx && cellCol <= endGridColIdx) // ? The column window check
				{
					cellIdx++;
					// * Cell index is incremental only
					// ? But since we make the instance list while parsing then the cells in the list come in order of their names.
					// ? This means that though the names would be in increasing order but it is not necessary for this to be true
					// ? in the sliding window. But, one thing is true that the cell index if not 0 then would be incremental only.
					// TODO :  Check that you are using cellIdx for calculating the pin index.
					(*instListItr)->cellIndex = cellIdx;
					/*gv.*/ cellsInWindow.push_back(*instListItr);
					// cout<<" and in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
				}
				else
				{
					(*instListItr)->cellIndex = 0; // Inside the row window but outside Window
				}
			}

			else
			{
				/*gv.*/ cellsBeyondRowWindow.push_back(*instListItr); // Outside the row window and window both
				(*instListItr)->cellIndex = 0;
				//cout<<" is NOT in row window "<<startGridRowIdx<<" "<<endGridRowIdx<<endl;
			}
			// ? We would have cellIdx = 0 for any cell that is beyond the current cliding window(cell not exixting inside cellsInRowWindow and cellsInWindow).
			/*
            if( cellCol >= startGridColIdx && cellCol <= endGridColIdx)
            {
                gv.cellsInColWindow.push_back(*instListItr);
                cout<<" is in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
            }
            else
            {
                gv.cellsBeyondColWindow.push_back(*instListItr);
                cout<<" is NOT in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
            }*/
		}
	}
	else // * If not for first time.
	{
		cout << "clearing cells in window list" << endl;
		/*gv.*/ cellsInWindow.clear(); // ? After each slide, the cellsInWindow needs to be emptied
		//		if(gv.cellsInRowWindow.empty())
		//		cout<<"Cells in Row Window list is empty"<<endl;
		for (itr = /*gv.*/ cellsInRowWindow.begin(); itr != /*gv.*/ cellsInRowWindow.end(); ++itr)
		{
			//	cout<<"Inside for loop"<<endl<<(*itr)->name;
			cellRow = (*itr)->row;
			cellCol = (*itr)->col;
			//cout<<"Cell row col accessed"<<endl;

			if (cellRow < startGridRowIdx || cellRow > endGridRowIdx)
			{
				(*itr)->cellIndex = 0; // ? Delete the cell from the row window if not in the new row bounds
									   //cout<<"Removing from cells in row window"<<endl;
									   //cout<<"Cell: "<<(*itr)->name<<endl;

				//gv.cellsInRowWindow.remove(*itr);
			}
			else
			{
				if (cellCol >= startGridColIdx && cellCol <= endGridColIdx)
				{
					cellIdx++; // * Again, the cell indexes are incremental
					(*itr)->cellIndex = cellIdx;
					//	cout<<"Adding cell in window"<<endl;
					/*gv.*/ cellsInWindow.push_back(*itr); // Put in the row window
				}
				else
				{
					(*itr)->cellIndex = 0; // ! DOUBT
				}
			}
		}
		if (startGridColIdx == /*gv.colBeginIdx*/ colStartIndex) //=> row window cell list to be updated as per new row window limits
		{														 // ! DOUBT
			//cout<<"=> row window cell list to be updated as per new row window limits"<<endl;

			for (itr = /*gv.*/ cellsBeyondRowWindow.begin(); itr != /*gv.*/ cellsBeyondRowWindow.end(); ++itr)
			{
				cellRow = (*itr)->row;
				cellCol = (*itr)->col;

				if (cellRow >= startGridRowIdx && cellRow <= endGridRowIdx) // ? Checking if the cell is in current row window
				{
					/*gv.*/ cellsInRowWindow.push_back(*itr);

					if (cellCol >= startGridColIdx && cellCol <= endGridColIdx)
					{
						cellIdx++;
						(*itr)->cellIndex = cellIdx;
						/*gv.*/ cellsInWindow.push_back(*itr);
					}
					else
					{
						(*itr)->cellIndex = 0;
					}
					//gv.cellsBeyondRowWindow.remove(*itr);
				}
			}
		}
	}
	// TODO : Check if the cell window being captured are correct or not

	cout << "Total number of cells in window: " << cellIdx << endl;
	if (cellIdx > 2000)
	{
		cout << "Returning from function as number of cells > 2000" << endl;
		return 0;
	}
	else if (cellIdx == 0)
	{
		cout << "Returning from function as there are no cells in the window" << endl;
		return 0;
	}

	//populate connectivity matrix for this window and generate data samples
	// ? ******************************************************* CELLS IN WINDOW CREATED, NOW PASSING TO MODEL STARTS ******************************************
	list<Inst *>::iterator adjCellItr; //, findItr;
	unsigned cellCount = 0, currentCellCount = 0, movableCellCount = 0;

	bool currentCellAccepted = false;
	bool cellPresentInCellIdxList = false;
	Inst *currentCell;
	unsigned numPins = 0;
	for (itr = /*gv.*/ cellsInWindow.begin(); itr != /*gv.*/ cellsInWindow.end(); ++itr) // Iterating the cells in window
	{
		currentCellAccepted = false; // Each time, current cell initially not accepted
		cellPresentInCellIdxList = false;
		currentCell = (*itr);
		unsigned currCellNumPins = currentCell->master->numMasterPins;
		//numPins += currCellNumPins;
		if (currentCell->cellIndex == 0)
		{ // ? Existing in the cellsInWindow but cell index = 0 then, skip this cell(Continue in the outermost for loop)
			cout << currentCell->name << " Cell index = 0 so continue in for loop" << endl;
			//numPins -= currCellNumPins ;
			continue; // !continue means skip the cell ->  this is not useful for handling more number of pins
		}

		if ((cellCount >= 2))
		{
			if (movableCellCount > 0) //if cells in list reached around 10
			{
				cout << "Cell count being passed : " << cellCount << endl;
				cout << "Cell: " << currentCell->name << " cell Index is: " << currentCell->cellIndex << " adjacency list size + cell Count " << cellCount << " is more than 10 in pre-check" << endl;
				/*gv.*/ k = cellCount;
				cout << "Movable Cell Count = " << movableCellCount << endl;
				cout << "Numpins in design  (Being sent for prediction) :" << numPins << endl;
				clearPinConnectivityMatrix(pin_connectivity_matrix);
				int flag = createPinConnectivityForDesign(pin_connectivity_matrix, cellIdxList, numPinsArray, pinIndexList, k); // This check is useless

				if (flag == -1)
				{
					cout << "#############################################" << endl;
					cout << "Warning 1 : Pins aree greater than 50 here." << endl;
					cout << "#############################################" << endl;
					/*cellCount = 0;
					currentCellCount = 0;
					movableCellCount = 0;																								 // After dumping, everything goes to default
					clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample

					movableCellList.clear();
					numPins = 0;
					continue;*/
				}
				cout << "\nCreating Output data sample\n";
				generateOutputDataSample(dataFile, rowStartIndex, rowEndIndex, colStartIndex, colEndIndex,
										 startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, pin_connectivity_matrix,
										 rowPosCell, colPosCell, cellIdxList, demandValCell, movableCellIdxList, numPinsArray, pinIndexList, movableCellList, k);
				/*
				* IDEA : When I am dumping the data, I have the cellIdxList array and all the cellindexes have been embedded inside the particular
				* instances. I want a pin connectivity matrix just for the pins that belong to the cells inside this cellIdxList.

			 	  TODO : Implement the following IDEA:-
				   * -> Go to all the cells in the cell index list one by one.
				   	 -> Go down the hierarchy their pinList.
					 -> Traverse the pin list and at each pin see its adjacency list and keep looking for pins that belong to the cells which exist in the
					    cellIdxList.
					 -> Now you have to make pin_connectivity for both of these.
			       * -> How to decide the index?
				   Are the indices in serial order ? --> This needs to be confirmed but it is obviously not true or necessary.
				   TODO : Indices need to be readjusted before passing.

				    -> One approach for pin connectivity:
					Use readjusted cell indices and use them to create the pin connecitivity matrix.

					? Important point: You might think that you might skip some or other cell index but cell count is continuous hence there will be no
					? zero in between different cell indices in the cellIdxList. Now, since we want to pass the pin connectivity according to what we -
					? have done in the data generation hence, is we mention the cellA in the row1 and col1 position then int the data, they are being c-
					? onsidered as the index 1 in the connectivity matrices. Since, while passing the design data to the ML model we are simply putting
					? the cells in order of the cell count. Hence we can create the pin_connectivity specific to count. This will lead to consistency b-
					? tween the data generated for training and the passed data.
				*/
				cellCount = 0;
				currentCellCount = 0;
				movableCellCount = 0;																								 // After dumping, everything goes to default
				clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample
				/*gv.*/ movableCellList.clear();
				numPins = 0;
			}
		}
		if (((/*cellCount +*/ (currentCell->adjacencyList.size())) < 10) /*&& (numPins + currCellNumPins < 50)*/) // ? Process of considering the current cell will go ahead
		{
			cellPresentInCellIdxList = isCellPresentInCellIdxList(currentCell->cellIndex, cellIdxList); // ! DOUBT : When was this cellIdxList created? -- > Being created continuously
			//cout<<"Cell: "<< currentCell->name<<" cell Index is: "<<currentCell->cellIndex<<" is present in CellIdxList? "<<cellPresentInCellIdxList<<endl;
			/*
			? About cellIdxList
				-> In this if block there is no possibility that the cell is out of window.
				-> But now we want to put the cells in the cellIdxList. So, simply we are checking if it already exists in the cellIdxList
				   and putting it in the list if not there.
			*/
			if (cellPresentInCellIdxList == false)
			{ // ? Not there in the list then put it. This is what is the intention behind making the function.
				cellCount++;
				/*gv.*/ rowPosCell[cellCount] = ((currentCell->row) - startGridRowIdx + 1);
				if (/*gv.*/ rowPosCell[cellCount] == 0)
				{
					cout << "ERROR : assigned rowPosCell as 0";
					//(0);
				}
				/*gv.*/ colPosCell[cellCount] = ((currentCell->col) - startGridColIdx + 1);
				if (/*gv.*/ colPosCell[cellCount] == 0)
				{
					cout << "ERROR : assigned colPosCell as 0";
					//(0);
				}
				/*gv.*/ numPinsArray[cellCount] = ((currentCell->master->numMasterPins));
				if (/*gv.*/ numPinsArray[cellCount] > 7)
				{
					cout << "numPinsArray Error 1: " << /*gv.*/ numPinsArray[cellCount] << endl;
					//(1);
				}
				/*gv.*/ demandValCell[cellCount] = getDemandsOfCell(currentCell);
				/*gv.*/ cellIdxList[cellCount] = currentCell->cellIndex; // Tells what is index of cell against the cell count
				// TODO : Change the pin connectivity according to this---> Do not try to change the current scheme but chnage this pin connectivity.
				currentCellCount = cellCount;		 //if needed to delete then from this position only as cell added first time in the list
				numPins = numPins + currCellNumPins; // # Change here if needed
				cout << "Cell count = " << cellCount << endl;
			}
			else
			{
				currentCellCount = cellCount + 1; //if needed to delete then from one position ahead as this cell is part of list already

				//numPins -= currCellNumPins ;
				/*
				Tf the cell already exists in the cell index list then, we do not need to add it but it is possible that the cell needs to be deleted from the
				cellIdxList then in that case we use a currentCellCount one ahead of the cellCount. cellCount is count of the cells that we are sure about.

				? We do not want to delete this one but if in the adjacency, even if one cell is out of the window then we will discard all the other cells that
				? connected to that.
				*/
			}
			cout << "Cell: " << currentCell->name << " cell Index is: " << currentCell->cellIndex << " adjacency loop start" << endl;
			for (adjCellItr = currentCell->adjacencyList.begin(); adjCellItr != currentCell->adjacencyList.end(); adjCellItr++)
			{
				if ((*adjCellItr) == currentCell) // Zero wirelength hence we need not consider anything here.
				{
					cout << "Cell connected to itself so continue in for loop" << endl;
					continue;
				}
				cout << "is connected to Cell: " << (*adjCellItr)->name << " whose cell Index is: " << (*adjCellItr)->cellIndex << endl;
				//findItr = std::find(gv.cellsInWindow.begin(), gv.cellsInWindow.end(), (*adjCellItr));
				//if(findItr!=gv.cellsInWindow.end())
				//find NOT WORKING with Inst* as the search element, works only with standard data type
				if ((*adjCellItr)->cellIndex != 0) //as only cells in window will have non zero cellIndex
				{								   // TODO: Since we are still considering the cell connectivity first, hence the cell adjacency has to be made based upon the pin adjacency.
												   // there is no reason why this should not be done
					cout << " and is in window";
					/*gv.*/ connectivity_matrix[currentCell->cellIndex][(*adjCellItr)->cellIndex] = 1;
					/*gv.*/ connectivity_matrix[(*adjCellItr)->cellIndex][currentCell->cellIndex] = 1;
					/*
					! DOUBT : What if this adjacent cell is ommited as we move ahead?
					*/
					/*
					Approach:-
					Traverse pinlist of the currenCell
						-> At each pin. Stop and traverse all the adjacent pins.

						-> Keep checking if the (*adjCellItr)->name == CurrentPin->inst->Name

						-> If yes then we have found the current pin and the adjacent corresponding pin.
						 		** Now, simply make pin_connectivity_matrix[][] = 1;
						* Calculation for index of any pin:-
						  TODO: pinsTillLastCell + Inst -> Pin -> pinIndexLocal
					*/

					// ?PinConnectionCreation:
					/*for (unordered_map<string, Pin *>::iterator i = currentCell->pinMap.begin(); i != currentCell->pinMap.end(); i++)
					{
						for (vector<Pin *>::iterator itr = (*i).second->adjacencyList.begin(); itr != (*i).second->adjacencyList.end(); itr++)
						{
							if (((*itr)->inst) == (*adjCellItr))
							{
								//cout << "\n\n++++++++I have matched two cells, now matching the pins+++++++++\n\n";
								// Calculating the pin index of pin of current cell
								unsigned pinsTillLastCell1 = 0;
								cout << "\n\n===>Finding index of pin : " << (*i).second->inst->name << "/P" << (*i).second->name << endl;
								for (unsigned j = 1; j < (*i).second->inst->cellIndex; j++)
								{
									Inst *temp1 = gv.dd.getInstanceFromCellIndex(j);
									assert(temp1);
									unsigned nopins = temp1->master->numMasterPins;
									//unsigned nopins = temp1->pinMap.size(); //temp1->master->numMasterPins;
									pinsTillLastCell1 += nopins;
									cout << nopins << "(" << (temp1->name) << ")"
										 << "+";
								}
								//(*itr)->pinIndexLocal
								cout
									<< (*i).second->pinIndexLocal << "(" << (*i).second->inst->name << ")"
									<< "=" << pinsTillLastCell1 + (*i).second->pinIndexLocal << endl;
								cout << "===>Finding index of pin : " << (*itr)->inst->name << "/P" << (*itr)->name << endl;

								unsigned pinsTillLastCell2 = 0;
								for (unsigned k = 1; k < (*itr)->inst->cellIndex; k++)
								{
									Inst *temp2 = gv.dd.getInstanceFromCellIndex(k);
									assert(temp2);
									unsigned nopins = temp2->master->numMasterPins;
									//unsigned nopins = temp2->pinMap.size(); //temp2->master->numMasterPins;
									pinsTillLastCell2 += nopins;
									cout << nopins << "(" << (temp2->name) << ")"
										 << "+";
								}
								cout << (*itr)->pinIndexLocal << "(" << (*itr)->inst->name << ")"
									 << "=" << pinsTillLastCell2 + (*itr)->pinIndexLocal << endl
									 << endl
									 << endl;
								gv.pin_connectivity_matrix[pinsTillLastCell1 + (*i).second->pinIndexLocal][pinsTillLastCell2 + (*itr)->pinIndexLocal] = 1;
								gv.pin_connectivity_matrix[pinsTillLastCell2 + (*itr)->pinIndexLocal][pinsTillLastCell1 + (*i).second->pinIndexLocal] = 1;

								if ((pinsTillLastCell1 + (*i).second->pinIndexLocal) > pinCount) //(pinsTillLastCell2 + (*itr)->pinIndexLocal)) // Greater of two being assigned
									pinCount = (pinsTillLastCell1 + (*i).second->pinIndexLocal);
								else if ((pinsTillLastCell2 + (*itr)->pinIndexLocal) > pinCount)
									pinCount = (pinsTillLastCell2 + (*itr)->pinIndexLocal);
							}
						}
					}*/
					cellPresentInCellIdxList = isCellPresentInCellIdxList((*adjCellItr)->cellIndex, cellIdxList);
					cout << " and present in CellIdxList? " << cellPresentInCellIdxList << endl;
					if (cellPresentInCellIdxList == false)
					{ // ? IF cell not present in the cellIdxList then add it to the cellIdxList
						cellCount++;
						/*gv.*/ cellIdxList[cellCount] = (*adjCellItr)->cellIndex;
						/*gv.*/ numPinsArray[cellCount] = ((*adjCellItr)->master->numMasterPins);
						if (/*gv.*/ numPinsArray[cellCount] > 20)
						{
							cout << "numPinsArray Error 1: " << /*gv.*/ numPinsArray[cellCount] << endl;
							//(1);
						}
						/*gv.*/ rowPosCell[cellCount] = (((*adjCellItr)->row) - startGridRowIdx + 1);
						if (/*gv.*/ rowPosCell[cellCount] == 0)
						{
							cout << "ERROR : assigned rowPosCell as 0";
							//(0);
						}
						/*gv.*/ colPosCell[cellCount] = (((*adjCellItr)->col) - startGridColIdx + 1);
						if (/*gv.*/ colPosCell[cellCount] == 0)
						{
							cout << "ERROR : assigned colPosCell as 0";
							//(0);
						}
						/*gv.*/ demandValCell[cellCount] = getDemandsOfCell(*adjCellItr);
					}
					if (cellCount > 10)
					{
						cout << "Error 1: cell count > 10 while in adjacency list loop, returning from function with 0" << endl;
						return 0;
					}

					numPins = numPins + currCellNumPins;
					currentCellAccepted = true; // This is done after doing everything else
				}
				else //adjacent cell not in window
				{
					/**
					  * ? If the connected cell is not inside window.
					  * ! DOUBT
					  * TODO : Resolve the doubt by looking at the code more carefully first
					  *
					  * Even if only one connected cell is out of the window then, the current cell will also be dumped and all other cells are not even going
					  * to be processed anyways.
					  *
					  * Video : 27 minutes
					  * */
					for (unsigned i = currentCellCount; i <= cellCount; i++)
					{
						/*gv.*/ cellIdxList[i] = 0;
						/*gv.*/ rowPosCell[i] = 0;
						/*gv.*/ demandValCell[i] = 0;
						/*gv.*/ colPosCell[i] = 0;

						numPins = numPins - numPinsArray[i];
						/*gv.*/ numPinsArray[i] = 0; //  ! BUG found here ->  we were doing nuPinsArray[cellCount] = 0
					}
					cellCount = currentCellCount - 1; // Already put cell was also deleted not just the adjacent.
					currentCellAccepted = false;
					cout << " is NOT in window, breaking out of adjacency list loop" << endl; // Reject all the connected cells and the current cell if out of window
					break;
				}
			}
			if (currentCellAccepted == true)
			{
				movableCellCount++; // ! DOUBT: Why compulsorily movable? -> Since we have not put this constraint yet
				/*gv.*/ movableCellIdxList[movableCellCount] = currentCell->cellIndex;
				/*gv.*/ movableCellList.push_back(currentCell);
			}
		}
		else if (movableCellCount > 0 /*|| (numPins > 50)*/) //? if cells in list reached around 10
		{
			cout << "Cell: " << currentCell->name << " cell Index is: " << currentCell->cellIndex << " adjacency list size + cell Count " << cellCount << " is more than 10" << endl;
			/*gv.*/ k = cellCount;
			cout << "Movable Cell Count = " << movableCellCount << endl;
			clearPinConnectivityMatrix(pin_connectivity_matrix);
			int flag = createPinConnectivityForDesign(pin_connectivity_matrix, cellIdxList, numPinsArray, pinIndexList, k);
			if (flag == -1)
			{
				cout << "#############################################" << endl;
				cout << "Error 2 : Pins cannot be greater than 50 here." << endl;
				cout << "#############################################" << endl;

				/*cellCount = 0;
				currentCellCount = 0;
				movableCellCount = 0;																								 // After dumping, everything goes to default
				clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample
				/*gv.*/
				//movableCellList.clear();
				//numPins = 0;
				//continue;*/
			}
			cout << "\nCreating Output data sample\n";
			generateOutputDataSample(dataFile, rowStartIndex, rowEndIndex, colStartIndex, colEndIndex, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, pin_connectivity_matrix, rowPosCell, colPosCell, cellIdxList,
									 demandValCell, movableCellIdxList, numPinsArray, pinIndexList, movableCellList, k);
			cellCount = 0;
			currentCellCount = 0;
			movableCellCount = 0;
			clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample
			numPins = 0;
			/*gv.*/ movableCellList.clear();
		}
		else
		{
			//cellCount--;
			cout << "Adjacency list of currentCell + cell count " << cellCount << " is > 10" << endl;
		}
	}

	if (movableCellCount > 0) //if cells in list reached around 10
	{
		cout << "After cell in windows list loop, Movable Cell Count = " << movableCellCount << endl;
		cout << "Cell count = " << cellCount << endl;
		/*gv.*/ k = cellCount;
		clearPinConnectivityMatrix(pin_connectivity_matrix);
		int flag = createPinConnectivityForDesign(pin_connectivity_matrix, cellIdxList, numPinsArray, pinIndexList, k);

		cout << "\nCreating Output data sample\n";
		if (1) //(flag == 1)
		{
			generateOutputDataSample(dataFile, rowStartIndex, rowEndIndex, colStartIndex, colEndIndex, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, pin_connectivity_matrix, rowPosCell, colPosCell, cellIdxList,
									 demandValCell, movableCellIdxList, numPinsArray, pinIndexList, movableCellList, k);
			numPins = 0;
			cellCount = 0;
			currentCellCount = 0;
			movableCellCount = 0;
			clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample
			/*gv.*/ movableCellList.clear();
		}
		numPins = 0;
		cellCount = 0;
		currentCellCount = 0;
		movableCellCount = 0;
		clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample
		/*gv.*/ movableCellList.clear();
	}
	if (pinCount > 0) //  JUST COUT
	{
		if (pinCount > 50)
		{
			cout << "Pins = " << pinCount << " ";
			//(3);
		}
		cout << endl
			 << "Pin connectivity matrix for " << pinCount << "pins:-" << endl;
		for (unsigned i = 1; i <= pinCount; i++)
		{
			//cout<< i << "-< ";
			for (unsigned j = 1; j <= pinCount; j++)
			{
				cout << /*gv.*/ pin_connectivity_matrix[i][j] << " ";
			}
			cout << "\n";
		}
		/*for (unsigned i = 1; i <= NUM_CELLS * 5; i++)
		{
			for (unsigned j = 1; j <= pinCount; j++)
			{
				cout << gv.pin_connectivity_matrix[i][j] << " ";
			}
			cout << "\n";
		}*/
	}
	else
		cout << endl
			 << "No pins were counted in the window, There is some problem with the pin_connectivity_matrix loop !!" << endl;
	return 1;
}

unsigned generateBitVectorOfCongestionMatrix(unsigned row, unsigned col)
{
	unsigned bitVector = 0;
	unsigned lay = 1;
	for (lay = 1; lay <= gv.l; lay++)
	{
		bitVector = bitVector << 1;
		if (gv.congestion_matrix[lay][row][col] == 1)
		{
			bitVector |= 1;
		}
		//cout<<gv.congestion_matrix[lay][row][col]<<" ";
	}
	//cout<<endl;
	//cout<<"Bit Vector: "<<bitVector<<endl;

	return bitVector;
}

void generateDataSample(ofstream &dataFile, unsigned movableCellIndex,
						unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx, unsigned compareDem,
						bool initialize, bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL], unsigned *rowPosCell, unsigned *colPosCell, unsigned *cellIdxList,
						unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k, float *X, float *Y)
{ // iniitalize variable is passed as first

	//PyObject *pValue2;
	unsigned connectivityRegister = 0x01;
	unsigned cellIndex = 0, connectedCellIndex = 0;
	unsigned bitVector;
	unsigned i = 1, j = 1;

	unsigned movedCellLocation = 0;

	unsigned inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL;

	std::memset(X, 0, sizeof(X));

	X[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = gv.n;
	//gv.pred[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = gv.n;
	//	pValue2 = PyLong_FromLong(X[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL]);
	// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL, pValue2);
	//cout<<"PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL ok"<<endl;
	X[PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL] = gv.m;
	//gv.pred[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = gv.m;
	//	pValue2 = PyLong_FromLong(X[PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL]);
	// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL, pValue2);

	//X[PYTHON_INPUT_VECTOR_NUM_LAYERS_INDEX_VAL] = gv.l;
	//gv.pred[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = gv.l;
	//	pValue2 = PyLong_FromLong(X[PYTHON_INPUT_VECTOR_NUM_LAYERS_INDEX_VAL]);
	// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, PYTHON_INPUT_VECTOR_NUM_LAYERS_INDEX_VAL, pValue2);

	X[PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL] = /*gv.*/ k;
	//gv.pred[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = /*gv.*/k;
	//	pValue2 = PyLong_FromLong(X[PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL]);
	// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL, pValue2);

	//dataFile<<gv.n<<","<<gv.m<<","<</*gv.*/k<<","<<gv.l<<",";
	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL;
	for (i = 1; i <= /*gv.*/ k; i++)
	{
		if ((initialize == false) && (/*gv.*/ cellIdxList[i] == movableCellIndex))
		{
			movedCellLocation = i;
		}
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ rowPosCell[i] : 0);
		//gv.pred[inputVectorCountML] = ((initialize == false) ? /*gv.*/rowPosCell[i] : 0);
		if ((/*gv.*/ rowPosCell[i] == 0) && (initialize == false))
		{
			cout << "ERROR : Row position cannot be zero.\nFound it to be zero at i = " << i << endl
				 << "Numcells = " << /*gv.*/ k << endl;
			//(0);
		}

		/*
		pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		pValue reference stolen here:
		PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);
		*/
		inputVectorCountML++;
		//dataFile<</*gv.*/rowPosCell[i]<<",";
	}

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_COLPOS_INDEX_VAL;
	for (j = 1; j <= /*gv.*/ k; j++)
	{
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ colPosCell[j] : 0);
		//gv.pred[inputVectorCountML] = ((initialize == false) ? /*gv.*/colPosCell[j] : 0);
		if ((/*gv.*/ colPosCell[j] == 0) && (initialize == false))
		{
			cout << "ERROR : Col position cannot be zero.\nFound it to be zero at j = " << j << endl
				 << "Numcells = " << /*gv.*/ k << endl;
			//(0);
		}

		/*
		pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		pValue reference stolen here:
		PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);
	 	*/
		inputVectorCountML++;
		//dataFile<</*gv.*/colPosCell[j]<<",";
	}

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_NUMPINS_INDEX_VAL;
	for (j = 1; j <= /*gv.*/ k; j++)
	{
		X[inputVectorCountML] = ((initialize == false) ? numPinsArray[j] : 0);
		//gv.pred[inputVectorCountML] = ((initialize == false) ? /*gv.*/numPinsArray[j] : 0);
		inputVectorCountML++;
	}

	return;

	//!!!!!!!!!!!!!!!!!!
	// ! ###################### HARD CODED ###########################################
	cout << "Started making pins vector" << endl;
	inputVectorCountML = PYTHON_INPUT_VECTOR_PIN_CONNECTIONS;

	if (GENERATE_SINGLE_CELL_CONNECTIVITY_IN_DATAFILE)
	{
		Inst *cell;
		unsigned firstPinId = 1, lastPinId = 1;
		for (int i = 1; i <= NUM_CELLS; i++)
		{
			if (cellIdxList[i] == movableCellIndex)
			{
				firstPinId = pinIndexList[i].low;
				lastPinId = pinIndexList[i].high;
			}
		}
		cout << "First and last pin ID :" << firstPinId << ", " << lastPinId << endl;
		unsigned p = 0;
		unsigned long long int decimalNumber = 0;
		int i = firstPinId;
		unsigned numBitsPerSplit = MAX_PINS_FOR_ANY_CELL * NUM_CELLS / NUM_SPLITS_PER_ROW;
		for (; i <= lastPinId; i++)
		{
			for (unsigned j = 1; j <= MAX_PINS_FOR_ANY_CELL * NUM_CELLS; j++)
			{
				if (p >= numBitsPerSplit)
				{
					X[inputVectorCountML] = decimalNumber;
					//cout << "Input vector index  :" << inputVectorCountML << endl;
					inputVectorCountML++;
					p = 0;
					decimalNumber = 0;
				}
				decimalNumber += pin_connectivity_matrix[i][j] * pow((double)2, (double)(numBitsPerSplit - p - 1));
				p++;
			}
		} // Let next index be decided by the input vector index itself
		  //cout << "Trying to get the cell";
		  //unsigned numberOfMasterPins = 0;
		  /*
		if(!initialize)
		{for (int i = 1; i < movableCellIndex; i++)
		{
			cell = gv.dd.getInstanceFromCellIndex(i);
			//cout << endl << "Got the cell " << cell->name;
			assert (cell);
			firstPinId += cell->master->numMasterPins;
		}}

		//unsigned lastPinId = 0 ;
		//cout << endl << "Going to find cell ID : " << movableCellIndex << " while initialize is " << initialize;

		if(!initialize)
		lastPinId = firstPinId + gv.dd.getInstanceFromCellIndex(movableCellIndex)->master->numMasterPins - 1;
		*/
		  /*
		for (int i = firstPinId; i <= lastPinId; i++)
		{
			//cout << "\nDecimal MSB1 : ";
			int k = 1;
			int dec = 0;
			unsigned p = 1;
			for (; k <= 56 / 4; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal LSB1 : ";
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 56 / 2; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal MSB2 : ";

			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 3 * 56 / 4; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal LSB2 : ";
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 50; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//out << "---->" << dec << endl;
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
		}*/
	}
	else
	{
		for (int i = 1; i <= 50; i++)
		{
			//cout << "\nDecimal MSB1 : ";
			int k = 1;
			int dec = 0;
			unsigned p = 1;
			for (; k <= 56 / 4; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += /*gv.*/ pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal LSB1 : ";
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 56 / 2; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += /*gv.*/ pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal MSB2 : ";

			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 3 * 56 / 4; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += /*gv.*/ pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//cout << "---->" << dec << "\nDecimal LSB2 : ";
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
			dec = 0;
			p = 1;
			for (; k <= 50; k++)
			{
				//	cout << gv.pin_connectivity_matrix[i][k];
				dec += /*gv.*/ pin_connectivity_matrix[i][k] * pow(2, 14 - p);
				p++;
			}
			//out << "---->" << dec << endl;
			X[inputVectorCountML] = float(dec) / PIN_CONNECTIVITY_SCALING_FACTOR;
			inputVectorCountML++;
		}
	}
	cout << "Ended making pins vector" << endl;
	// ! ###################### HARD CODED END ###########################################

	/*inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_NUMPINS_INDEX_VAL;
	for (j = 1; j <= gv.k; j++)
	{
		X[inputVectorCountML] = ((initialize == false) ? gv.numPinsArray[j] : 0);
		inputVectorCountML++;
	}*/

	// Pin connectivity:-

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_DEMAND_INDEX_VAL;
	for (j = 1; j <= /*gv.*/ k; j++)
	{
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ demandValCell[j] : 0);

		inputVectorCountML++;
		//dataFile<</*gv.*/colPosCell[j]<<",";
	}

	inputVectorCountML = PYTHON_INPUT_VECTOR_SUPPLY_DEMAND_VAL_INDEX_VAL;
	for (unsigned lay = 1; lay <= gv.l; lay++)
	{
		for (i = startGridRowIdx; i <= endGridRowIdx; i++)
		{
			for (j = startGridColIdx; j <= endGridColIdx; j++)
			{
				if (initialize == false)
				{
					if (lay > NUM_LAYERS_IN_MODEL)
						goto label1;
					X[inputVectorCountML] = ((float)(gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j])) / SUP_DEM_SCALING_FACTOR; //  change has been done here

					/*if (i == 6 && j == 1)
					{
						cout << "\nAt 6,1 ---->" << X[inputVectorCountML] << endl; //inputVectorCountML++;
						cout << "Supply Dem = " << gv.dd.gGrid_supply[lay][i][j] << " " << gv.dd.gGrid_demand[lay][i][j] << endl;
					} */
					//X[inputVectorCountML] = gv.dd.gGrid_demand[lay][i][j];
				}
				inputVectorCountML++;
			}
		}
	}

label1:
	/*inputVectorCountML = PYTHON_INPUT_VECTOR_CONGESTION_INDEX_VAL;
    for(i=startGridRowIdx;i<=endGridRowIdx;i++)
    {
        for(j=startGridColIdx;j<=endGridColIdx;j++)
        {
            if(initialize == false)
            {
                bitVector = generateBitVectorOfCongestionMatrix(i,j);
                X[inputVectorCountML] = bitVector;

		//pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);

               inputVectorCountML ++ ;
            }
            else
            {
                X[inputVectorCountML] = 0;

	//	pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
	//	PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);

                inputVectorCountML ++ ;
            }
            //dataFile<<bitVector<<",";
        }
    }
*/
	/*inputVectorCountML = PYTHON_INPUT_VECTOR_CONNECTIVITY_INDEX_VAL;

	for (i = 1; i <= gv.k; i++)
	{
		connectivityRegister = 0x01;
		cellIndex = gv.cellIdxList[i]; // Picked a cell index (not cell count) -- > But this is correct.
		for (j = 1; j <= gv.k; j++)
		{
			connectedCellIndex = gv.cellIdxList[j]; // all the others being considered as connected cell
			connectivityRegister = connectivityRegister << 0x01;
			connectivityRegister |= (gv.connectivity_matrix[cellIndex][connectedCellIndex]) ? 1 : 0;
			//X[inputVectorCountML] = ( (initialize==false)? gv.connectivity_matrix[gv.cellIdxList[i]][gv.cellIdxList[j]] : (-1) );


		//pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
		//PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);

			//           inputVectorCountML ++ ;
			//dataFile<<gv.connectivity_matrix[gv.cellIdxList[i]][gv.cellIdxList[j]]<<",";
		}
		X[inputVectorCountML] = ((initialize == false) ? float(connectivityRegister) / CONNECTIVITY_SCALING_FACTOR : 0);
		//inputVectorCountML++;
		//dataFile<<connectivityRegister<<",";
	}*/

	inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_INDEX_INDEX_VAL;

	X[inputVectorCountML] = ((initialize == false) ? movedCellLocation : 0);

	/*
		pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
		PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);
	*/
	//inputVectorCountML ++ ;
	//dataFile<<movedCellLocation<<",";

	inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_ROWPOS_INDEX_VAL;
	X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ rowPosCell[movedCellLocation] : 0);

	/*
		pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
		PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);
	*/
	//inputVectorCountML ++ ;
	//dataFile<</*gv.*/rowPosCell[movedCellLocation]<<",";
	inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_COLPOS_INDEX_VAL;
	X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ colPosCell[movedCellLocation] : 0);
	/*
		pValue2 = PyLong_FromLong(X[inputVectorCountML]);
		// pValue reference stolen here:
		PyList_SetItem(pModelPredictFuncArgsObj, inputVectorCountML, pValue2);
	*/
	//cout<<"PYTHON_INPUT_VECTOR_MCELL_COLPOS_INDEX_VAL ok"<<endl;
	inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_DEMAND_INDEX_VAL;
	X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ demandValCell[movedCellLocation] : 0);

	inputVectorCountML++;
	//dataFile<</*gv.*/colPosCell[movedCellLocation]<<",";
	if (inputVectorCountML != 248)
	{
		// logs << "Error: inputVectorCount = " << inputVectorCountML;
		;
	} //
	else
		// logs << endl
		// 	 << "Successfully created the data sample";
		;
	//    doing endl after passing predicted values
	//dataFile<<endl;
}
void clearDataSample(float *X)
{
	for (int i = 0; i <= PYTHON_INPUT_VECTOR_MCELL_DEMAND_INDEX_VAL; i++)
		X[i] = 0;
}
void generateOutputDataSample(ofstream &dataFile, unsigned rowStartIndex, unsigned rowEndIndex, unsigned colStartIndex, unsigned colEndIndex,
							  unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
							  bool pin_connectivity_matrix[NUM_CELLS * MAX_PINS_FOR_ANY_CELL][NUM_CELLS * MAX_PINS_FOR_ANY_CELL], unsigned *rowPosCell, unsigned *colPosCell, unsigned *cellIdxList, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray,
							  pinIndexes *pinIndexList, list<Inst *> &movableCellList, unsigned k)
{
	double totalTime = 0;
	ofstream pMoveLogs;

	float X[248];
	float Y[2];

	//pMoveLogs.open("predictionMovements.txt", ios::app); // ios::app allows you to write anywhere in file(not just in the end)
	pMoveLogs.close(); //taking lot of disk space
	unsigned i = 1;
	unsigned rowGridStart = 1, colGridStart = 1, rowGridEnd = 10, colGridEnd = 10, mCellGridRow = 1, mCellGridCol = 1;
	auto timebefore = chrono::system_clock::to_time_t(chrono::system_clock::now());
	pMoveLogs << "generateOutputDataSample func started at : " << ctime(&timebefore) << endl;
	list<Inst *>::iterator itr = /*gv.*/ movableCellList.begin();
	//Initializing the X vector
	//generateDataSample(dataFile, 0, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, true);
	clearDataSample(X);
	/*
	
	*/
	std::unordered_set<string> pinNetList;
	for (itr = /*gv.*/ movableCellList.begin(); itr != /*gv.*/ movableCellList.end(); itr++)
	{

		//if ((*itr)->cellIndex != 0) //: Commented for unit testing, passing cell3 only (one sample case)
		if ((*itr)->cellIndex == FIXED_CELL_IDX)
		{
			//<<(*itr)->name<<",";
			//<<startGridRowIdx<<","<<startGridColIdx<<",";
			//std::unordered_set<string> pinNetList;
			/*gv.*/ pinNetList.clear();
			unsigned totNetDemand = 0;
			//totDemand += (*itr)->master->; // Demand is not so simple here-- > You have adjacent cell and other demand constraints too.
			cout << endl
				 << "Cell : " << (*itr)->name;
			for (unordered_map<string, Pin *>::iterator pItr = (*itr)->pinMap.begin(); pItr != (*itr)->pinMap.end(); pItr++)
			{
				cout << endl
					 << "Net : " << (*pItr).second->net->name;
				if (/*gv.*/ pinNetList.find((*pItr).second->net->name) == /*gv.*/ pinNetList.end())
				{

					totNetDemand++;
					/*gv.*/ pinNetList.insert((*pItr).second->net->name);
				}
			}
			cout << "\nTotal net demand for the cell to be moved is = " << totNetDemand;
			unsigned blockageL1 = 0;
			for (vector<Blockage>::iterator bitr = (*itr)->master->blockageList.begin(); bitr != (*itr)->master->blockageList.end(); bitr++)
			{
				if ((*bitr).layer == 1)
				{
					blockageL1 = (*bitr).demand;
					break;
				}
			}
			if (maxNetsPerCell < totNetDemand)
			{
				maxNetsPerCell = totNetDemand;
			}
			cout << endl
				 << "Total Blockage for the cell in L1 layer : " << blockageL1;
			cout << endl
				 << "Blockage demand + net demand = " << totNetDemand + blockageL1;

			generateDataSample(dataFile, /*gv.*/ movableCellIdxList[i] /* FIXED_CELL_IDX*/, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, /*totNetDemand +*/ blockageL1,
							   false, pin_connectivity_matrix, rowPosCell, colPosCell, cellIdxList, demandValCell, movableCellIdxList, numPinsArray, pinIndexList, k, X, Y);

			// ! generateDataSample uses X but we should ow use 8 different X[]
			//float X_trial[] = {10, 10, 10, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1, 5, 7, 5, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16777220, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 1, 3, 1, 5, 5, 3, 0, 7, 5, 4, 10, 0, 1, 1, 0, 0, 4, 3, 3, 10, 0, 2, 8, 0, 7, 6, 4, 0, 0, 6, 10, 10, 2, 4, 2, 5, 0, 10, 10, 1, 0, 0, 0, 4, 1, 5, 8, 1, 2, 5, 2, 10, 7, 0, 1, 2, 5, 2, 10, 7, 10, 0, 7, 4, 0, 6, 3, 2, 0, 8, 3, 0, 2, 0, 5, 10, 1, 0, 8, 10, 0, 0, 0, 2, 0, 10, 0, 2, 6, 0, 0, 4, 0, 6, 0, 6, 10, 3, 8, 9, 0, 3, 0, 2, 4, 1, 3, 4, 1, 1, 1};
			//pythonPredictionCode(X,Y);
			//pythonRandomPrediction();
			cout << "Predicted values are: " << round(Y[0]) << " and " << round(Y[1]) << endl;
			predictionLogs << round(Y[0]) << "," << round(Y[1]) << ",";
			predictionLogs << (*itr)->name << ",";
			//<<Y[0]<<","<<Y[1]<<",";
			//<<endl;
			numPredicitionCalls++;

			if ((round(Y[0]) > 10) || (round(Y[0]) < 1))
				Y[0] = printRandoms(1, 10);

			if ((round(Y[1]) > 10) || (round(Y[1]) < 1))
				Y[1] = printRandoms(1, 10);

			rowGridStart = rowStartIndex; // gv.rowBeginIdx;
			rowGridEnd = rowEndIndex;	  // gv.rowEndIdx;
			colGridStart = colStartIndex; //  gv.colBeginIdx;
			colGridEnd = colEndIndex;	  // gv.colEndIdx;

			mCellGridRow = round(Y[0]) + startGridRowIdx - 1;
			mCellGridCol = round(Y[1]) + startGridColIdx - 1;

			unsigned nMCells = 0;

			unsigned tempMCellGridRow = mCellGridRow;
			unsigned tempMCellGridCol = mCellGridCol;
			unsigned restoreRow = mCellGridRow;
			unsigned restoreCol = mCellGridCol;
			unsigned numTrials = 0;

		label_repeat:
			unsigned tempNMCells = gv.numcellsmoved;

		//	nMCells = pMoveCell((*itr), rowGridStart, rowGridEnd, colGridStart, colGridEnd, tempMCellGridRow, tempMCellGridCol, pMoveLogs);
			tempMCellGridRow = restoreRow;
			tempMCellGridCol = restoreCol;
			if (0) //(tempNMCells == nMCells)
			{
				cout << endl
					 << "Trial number " << numTrials + 1 << " for particular cell";
				numTrials++;
				//numTrials = 9; // added to remove the repeated trials
				switch (numTrials)
				{
				case 1:
					tempMCellGridRow++;
					break;
				case 2:
					tempMCellGridCol++;
					break;
				case 3:
					tempMCellGridRow--;
					break;
				case 4:
					tempMCellGridCol--;
					break;
				case 5:
					tempMCellGridRow++;
					tempMCellGridCol++;
					break;
				case 6:
					tempMCellGridRow--;
					tempMCellGridCol++;
					break;
				case 7:
					tempMCellGridRow--;
					tempMCellGridCol--;
					break;
				case 8:
					tempMCellGridRow++;
					tempMCellGridCol--;
					break;
				default:
					goto label_skip_trial;
					break;
				}
				goto label_repeat;
			}
		label_skip_trial:
// phon
			cout << "If moved then moved to " << tempMCellGridRow << " " << tempMCellGridCol << endl;
			if (i > 10)
				cout << "Error 2: MovableCellIdxList index >10" << endl;
			i++;
			numTotalTrials += numTrials;

			// auto now = std::chrono::high_resolution_clock::now();
			// std::chrono::duration<double, std::ratio<60>> elapsed = now - start;
			// totalTime += elapsed.count();
			// cout << "Elapsed time is : " << elapsed.count();
			// if (elapsed.count() > 180)
			// {

			// 	ofstream timingLogs("timing_logs_new.txt"+outFile);

			// 	timingLogs << "Did not complete the scan 1 itself even after fixed time" << endl;
			// 	timingLogs << "Number of times repeated : " << numRepeat << endl;
			// 	timingLogs << "\nDuration of running :" << elapsed.count() << "m" << endl;
			// 	//timingLogs << "\nWirelength reduced by " << totWLReduction << ", hence producing output for check." << endl;
			// 	timingLogs << "Number of times repeated : " << numRepeat << endl;
			// 	timingLogs << "Maximum nets per cell : " << maxNetsPerCell << endl;
			// 	cout << "Maximum Nets per cell = " << maxNetsPerCell << endl;
			// 	cout << "numMovedCells = " << gv.numcellsmoved << endl;
			// 	cout << "numPredictionTrials = " << numPredicitionCalls << endl;

			// 	produceRouterLogs();
			// 	gv.dd.produceOutput(outFile);
			// 	exit(0);
			// 	//break;
			// }
			//initialize input vector X for further iterations
			//clearCellArrays();
			//generateDataSample(dataFile, 0, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, true);
		}
	}
	auto timeafter = chrono::system_clock::to_time_t(chrono::system_clock::now());
	pMoveLogs << "generateOutputDataSample funct finished at : " << ctime(&timeafter) << endl;
}

int slidingWindowScan(vector<Inst *> &instList, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex, unsigned rowSize, unsigned colSize, unsigned rowStep, unsigned colStep,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials,const fdeep::model &model)
{
//    /* for(int i=0;i<1000;i++){
// 		numPredictionTrials++;
// 	}*/
// 	//cout << "Start"<< rowStartIndex << endl;
// 	// rowSize and colSize are sliding window sizes
// 	// rowStartIndex, colStartIndex, rowEndIndex, colEndIndex define the size of the particular partition to run sliding window upon.
// 	// rowStep and colStep have thier usual meaning
 	ofstream dataFile;
// 	//    char ch = 'N';
// 	//dataFile.open("dataFile.csv");

 	list<Inst *> cellsInWindow;
 	list<Inst *> cellsInRowWindow;
 	list<Inst *> cellsBeyondRowWindow;

 	/*gv.*/ cellsInWindow.clear(); // ! Create separate
 	/*gv.*/ cellsInRowWindow.clear();
 	/*gv.*/ cellsBeyondRowWindow.clear();
 	unsigned startGridRowIdx = rowStartIndex, endGridRowIdx = startGridRowIdx +rowSize-1, startGridColIdx = colStartIndex, endGridColIdx = startGridColIdx + colSize -1; // these are for sliding
 	unsigned rowIncr = rowStep, colIncr = colStep;
	//cout << "Row Step: " << rowStep << endl;

	// while(difftime(time(NULL), start) < 120){
    // 	numPredictionTrials += 1;
    // }
	// return 1;

	while (rowIncr != 0)
	{
		//cout<<"Row Window: "<<startGridRowIdx<<" "<<endGridRowIdx<<endl;
		//cout << "RowIncr" << endl;
		while (colIncr != 0)
		{
			//cout <<"Col Window: "<<startGridColIdx<<" "<<endGridColIdx<<endl;
			//cout << "ColIncr" << endl;
			int flag = partitionDesignClustered(instList, dataFile, rowStartIndex, colStartIndex, rowEndIndex, colEndIndex, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, cellsInWindow, cellsInRowWindow, cellsBeyondRowWindow,logs,numCellsMoved,numPredictionTrials, model); 
			if(flag==-1) return 1;
			if (flag == 1)
			{
				logs << "partition successful and output generated" << endl;
				//generateOutputDataSample(dataFile,startGridRowIdx,endGridRowIdx,startGridColIdx,endGridColIdx);
			}
			if (flag == 0) return 1;
			startGridColIdx += colIncr;
			endGridColIdx = (startGridColIdx + colSize - 1);
			if (endGridColIdx > colEndIndex)
			{
				colIncr = 0;
			}
		}
		colIncr = colStep;
		startGridColIdx = colStartIndex;
		endGridColIdx = startGridColIdx + colSize - 1;
		/*cout<<"Continue Next row window? (Y/N)"<<endl;
        cin>>ch;
        if(ch == 'N' || ch == 'n')
                return 0;
		*/
		startGridRowIdx += rowIncr;
		endGridRowIdx = (startGridRowIdx + rowSize - 1);

		if (endGridRowIdx > rowEndIndex)
		{
			rowIncr = 0;
			startGridRowIdx = rowStartIndex;
			// startGridColIdx = colStartIndex;
		}
	}

	return 1;
}

void clearCellArrays(unsigned *cellIdxList, unsigned *rowPosCell, unsigned *colPosCell, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList) //  To be changed for number of pins above 50 in the design
{
	for (unsigned i = 1; i <= NUM_CELLS; i++)
	{
		/*gv.*/ cellIdxList[i] = 0;
		/*gv.*/ rowPosCell[i] = 0;
		/*gv.*/ colPosCell[i] = 0;
		pinIndexList[i].high = 0;
		pinIndexList[i].low = 0;
		/*gv.*/ demandValCell[i] = 0;
		/*gv.*/ movableCellIdxList[i] = 0;
		/*gv.*/ numPinsArray[i] = 0;
		/*gv.*/ cellIdxList[i] = 0;
		/*gv.*/ rowPosCell[i] = 0;
		/*gv.*/ colPosCell[i] = 0;
		/*gv.*/ demandValCell[i] = 0;
		/*gv.*/ movableCellIdxList[i] = 0;
		/*gv.*/ numPinsArray[i] = 0;
	}
	/*
	for (unsigned i = 1; i <= 50; i++)
	{
		for (unsigned j = 1; j <= 50; j++)
			gv.pin_connectivity_matrix[i][j] = 0;
	}*/
	//for(list<Inst *>::iterator itr = /*gv.*/movableCellList.begin(); itr!= /*gv.*/movableCellList.end(); itr++)
	//(*itr)->cellIndex = 0;
}

bool isCellPresentInCellIdxList(unsigned cellIdx, unsigned *cellIdxList)
{
	bool cellPresent = false;

	for (unsigned i = 1; ((i <= NUM_CELLS) && (/*gv.*/ cellIdxList[i] != 0)); i++)
	// TODO: Check if the condition cellIdxList[i] != 0 is valid or not by printing the cellIdxList at correct intervals in the code
	{
		if (/*gv.*/ cellIdxList[i] == cellIdx)
		{
			cellPresent = true;
			break;
		}
		else
			cellPresent = false;
	}

	return cellPresent;
}

int pythonPredictionCode(float* X, float* Y,const auto &model)
{
	//if(localModel) const auto model = fdeep::load_model("fdeep_model.json");

	//const auto os = model.get_output_shapes();

	// tensor input = fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(248)), std::vector<float>{X,X+248});
	// tensors inputs;
	// inputs.push_back(input);

	// const auto result = model.predict(inputs);

	const auto result = model.predict(
    {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(248)),
    std::vector<float>{X,X+248})});
 	std::vector<float> vec = result.data()->to_vector();
	// Y[0] = 0;
	// Y[1] = 1;
	int idx=0;
	for(auto & element : vec){
		Y[idx] = element;
		idx++;
	}
	return 0;
}

/*Returns the HPWL of the cell with its current location*/
unsigned hpwlForCell(Inst *Cell)
{
	unsigned totalhpwl = 0, nhpwl = 0; //netHPW
	Net *prevNet = NULL;
	for (vector<Net *>::iterator n = Cell->netList.begin();
		 n != Cell->netList.end(); ++n)
	{
		if ((*n) != prevNet)
		{
			prevNet = (*n);
			nhpwl = (*n)->hpwl();
			totalhpwl += nhpwl;
		}
	}
	//logs << "Total HPWL associated with cell : " << totalhpwl << endl;

	return totalhpwl;
}

/*Returns the delta HPWL if the cell is moved to the given location*/
int deltahpwlForCell(Inst *CelltoMove, unsigned xpos, unsigned ypos,
					 unsigned &newtotalhpwl, ofstream &logs)
{
	unsigned totalhpwl = 0;
	//unsigned newtotalhpwl = 0;
	unsigned cellrow, cellcol;

	//Store row and col of Cell to be Moved for restoring later;
	cellrow = CelltoMove->row;
	cellcol = CelltoMove->col;

	totalhpwl = hpwlForCell(CelltoMove);
	logs << "Total HPWL before cell " << CelltoMove->name << " moved: "
		 << totalhpwl << endl;

	logs << "In delta HPWL for cell function, (x,y): " << xpos << " " << ypos
		 << endl;
	/*	if (totalhpwl == 0) {
	 newtotalhpwl = 0;
	 } else {*/
	//Change row and col of Cell to be Moved for estimation;
	CelltoMove->row = xpos;
	CelltoMove->col = ypos;

	newtotalhpwl = hpwlForCell(CelltoMove);
	logs << "Total HPWL after cell " << CelltoMove->name << " moved: "
		 << newtotalhpwl << endl;
	//Restoring the row and col of Cell to original value
	CelltoMove->row = cellrow;
	CelltoMove->col = cellcol;
	//}
	return (newtotalhpwl - totalhpwl);
}

unsigned wirelengthForCell(Inst *Cell, ofstream &logs)
{
	unsigned totalwl = 0, netwl;
	Net *prevNet = NULL;
	for (vector<Net *>::iterator n = Cell->netList.begin();
		 n != Cell->netList.end(); ++n)
	{
		if ((*n) != prevNet)
		{
			prevNet = (*n);
			netwl = netWirelengthDemand(*n, false, false);
			totalwl += netwl;
		}
	}
	logs << "Total wirelength associated with cell  " << Cell->name << " is: "
		 << totalwl << endl;

	return totalwl;
}

void movedCellBlockDemand(Inst *Cell, unsigned row, unsigned col)
{
	for (vector<Blockage>::iterator j = Cell->master->blockageList.begin();
		 j != Cell->master->blockageList.end(); ++j)
	{

		m.lock();
		gv.dd.gGrid_demand[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		gv.dd.gGrid_demand[(*j).layer][row][col] -= (*j).demand; 
		m.unlock();
		// unsigned partitionNumber_a = whichPartition(Cell->row);
		// unsigned partitionNumber_b = whichPartition(row);
		// if(partitionNumber_a==1){
		// 	gv.dd.gGrid_demand_firstPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==2){
		// 	gv.dd.gGrid_demand_secondPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==3){
		// 	gv.dd.gGrid_demand_thirdPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==4){
		// 	gv.dd.gGrid_demand_fourthPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==5){
		// 	gv.dd.gGrid_demand_fifthPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==6){
		// 	gv.dd.gGrid_demand_sixthPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==7){
		// 	gv.dd.gGrid_demand_seventhPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }else if(partitionNumber_a==8){
		// 	gv.dd.gGrid_demand_eighthPartition[(*j).layer][Cell->row][Cell->col] += (*j).demand;
		// }

		// if(partitionNumber_b==1){
		// 	gv.dd.gGrid_demand_firstPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==2){
		// 	gv.dd.gGrid_demand_secondPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==3){
		// 	gv.dd.gGrid_demand_thirdPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==4){
		// 	gv.dd.gGrid_demand_fourthPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==5){
		// 	gv.dd.gGrid_demand_fifthPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==6){
		// 	gv.dd.gGrid_demand_sixthPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==7){
		// 	gv.dd.gGrid_demand_seventhPartition[(*j).layer][row][col] -= (*j).demand;
		// }else if(partitionNumber_b==8){
		// 	gv.dd.gGrid_demand_eighthPartition[(*j).layer][row][col] -= (*j).demand;
		// }
		
	}
}

void cellExtraDemand(unsigned x, unsigned y, bool removeDemand)
{
	unsigned sameGridDemand = 0;
	unsigned adjGridDemand = 0;

	//unsigned x = 1, y = 1;

	//Note that cellName variables actually store only the pointer and not the name of the cell
	MasterCell *cellName1;
	MasterCell *cellName2;

	unsigned cell1 = 0, cell2 = 0, cellPairs = 0; //counter for cells present in a list
	unsigned cellPairPre = 0, cellPairNext = 0;
	unsigned cell1Current = 0, cell2Current = 0;
	unsigned cell1Previous = 0, cell2Previous = 0;
	unsigned cell1Next = 0, cell2Next = 0;

	vector<MasterCell *> cellsInCurrentGrid;
	vector<MasterCell *> cellsInNextGrid;
	vector<MasterCell *> cellsInPrevGrid;

	////	unordered_map<string, int> netCovered;
	//	for (x = 1; x <= gv.rowEndIdx; x++) {
	//		for (y = 1; y <= gv.colEndIdx; y++) {
	cellsInCurrentGrid.clear();
	cellsInPrevGrid.clear();
	cellsInNextGrid.clear();

	for (vector<Inst *>::iterator i = gv.dd.instList.begin();
		 i != gv.dd.instList.end(); ++i)
	{
		if ((*i)->row == x && (*i)->col == y)
		{ //can use nested if to check y==col
			cellsInCurrentGrid.push_back((*i)->master);
		}
		else if ((*i)->row == x && (*i)->col == y - 1)
		{
			cellsInPrevGrid.push_back((*i)->master);
		}
		else if ((*i)->row == x && (*i)->col == y + 1)
		{
			cellsInNextGrid.push_back((*i)->master);
		}
	}

	for (vector<ExtraDemand *>::iterator i = gv.dd.extraDemandList.begin();
		 i != gv.dd.extraDemandList.end(); ++i)
	{

		cell1 = 0;
		cell2 = 0;
		cellPairPre = 0;
		cellPairNext = 0;
		cell1Current = 0;
		cell2Current = 0;
		cell1Previous = 0;
		cell2Previous = 0;
		cell1Next = 0;
		cell2Next = 0;

		cellName1 = (*i)->cell1;
		cellName2 = (*i)->cell2;

		if ((*i)->same == true)
		{

			getNoCellInGridCount(cellName1, cellName2, cellsInCurrentGrid,
								 cell1, cell2);
			//					int cell2 = getNoCellInGridCount(cellName2,
			//							cellsInCurrentGrid);
			cellPairs = (cell1 > cell2) ? cell2 : cell1;

			sameGridDemand = (cellPairs * (*i)->demand);
			//		gv.dd.gGrid_demand[(*i)->layer->number][x][y] += removeDemand?((-1)*sameGridDemand):sameGridDemand;
			unsigned partitionNumber = whichPartition(x);
			if (removeDemand == true)
			{
				if (gv.dd.gGrid_demand[(*i)->layer->number][x][y] != 0)
				{
					m.lock();
					gv.dd.gGrid_demand[(*i)->layer->number][x][y] -=
						sameGridDemand;
					m.unlock();
				}
				
				// if(partitionNumber==1){
				// 	if (gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==2){
				// 	if (gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==3){
				// 	if (gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==4){
				// 	if (gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==5){
				// 	if (gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==6){
				// 	if (gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==7){
				// 	if (gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }else if(partitionNumber==8){
				// 	if (gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] -= sameGridDemand;
				// 	}
				// }
			}
			else
			{
				m.lock();
				gv.dd.gGrid_demand[(*i)->layer->number][x][y] += sameGridDemand;
				m.unlock();
				// if(partitionNumber==1){
				// 	gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==2){
				// 	gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==3){
				// 	gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==4){
				// 	gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==5){
				// 	gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==6){
				// 	gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==7){
				// 	gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }else if(partitionNumber==8){
				// 	gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] += sameGridDemand;
				// }

			}
			sameGridDemand = 0;
		}

		else if ((*i)->same == false)
		{

			if (cellName1 == cellName2)
			{

				cell1Current = getNoCellInGridCount(cellName1,
													cellsInCurrentGrid);
				cell1Previous = getNoCellInGridCount(cellName1,
													 cellsInPrevGrid);
				cell1Next = getNoCellInGridCount(cellName1, cellsInNextGrid);

				cellPairPre =
					(cell1Current > cell1Previous) ? cell1Previous : cell1Current;
				cellPairNext =
					(cell1Current > cell1Next) ? cell1Next : cell1Current;
			}
			else
			{
				getNoCellInGridCount(cellName1, cellName2, cellsInCurrentGrid,
									 cell1Current, cell2Current);
				getNoCellInGridCount(cellName1, cellName2, cellsInPrevGrid,
									 cell1Previous, cell2Previous);
				getNoCellInGridCount(cellName1, cellName2, cellsInNextGrid,
									 cell1Next, cell2Next);

				/////////////for finding cell pairs /////////////
				unsigned a =
					(cell1Current > cell2Previous) ? cell2Previous : cell1Current;
				unsigned b =
					(cell2Current > cell1Previous) ? cell1Previous : cell2Current;
				unsigned c =
					(cell1Current > cell2Next) ? cell2Next : cell1Current;
				unsigned d =
					(cell2Current > cell1Next) ? cell1Next : cell2Current;

				cellPairPre = a + b;
				cellPairNext = c + d;
			}
			adjGridDemand += (cellPairPre + cellPairNext) * (*i)->demand;
			//gv.dd.gGrid_demand[(*i)->layer->number][x][y] +=
			//removeDemand?((-1)*adjGridDemand):adjGridDemand;;
			unsigned partitionNumber = whichPartition(x);
			if (removeDemand == true)
			{
				if (gv.dd.gGrid_demand[(*i)->layer->number][x][y] != 0)
				{
					m.lock();
					gv.dd.gGrid_demand[(*i)->layer->number][x][y] -=
						adjGridDemand;
					m.unlock();
				}
				// if(partitionNumber==1){
				// 	if (gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==2){
				// 	if (gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==3){
				// 	if (gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==4){
				// 	if (gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==5){
				// 	if (gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==6){
				// 	if (gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==7){
				// 	if (gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }else if(partitionNumber==8){
				// 	if (gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] != 0)
				// 	{
				// 		gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] -= adjGridDemand;
				// 	}
				// }
			}
			else
			{
				m.lock();
				gv.dd.gGrid_demand[(*i)->layer->number][x][y] += adjGridDemand;
				m.unlock();
				// if(partitionNumber==1){
				// 	gv.dd.gGrid_demand_firstPartition[(*i)->layer->number][x][y] += adjGridDemand;
				// }else if(partitionNumber==2){
				// 	gv.dd.gGrid_demand_secondPartition[(*i)->layer->number][x][y] += adjGridDemand;
				// }else if(partitionNumber==3){
				// 	gv.dd.gGrid_demand_thirdPartition[(*i)->layer->number][x][y] +=adjGridDemand;
				// }else if(partitionNumber==4){
				// 	gv.dd.gGrid_demand_fourthPartition[(*i)->layer->number][x][y] += adjGridDemand;
				// }else if(partitionNumber==5){
				// 	gv.dd.gGrid_demand_fifthPartition[(*i)->layer->number][x][y] += adjGridDemand;
				// }else if(partitionNumber==6){
				// 	gv.dd.gGrid_demand_sixthPartition[(*i)->layer->number][x][y] += adjGridDemand;
				// }else if(partitionNumber==7){
				// 	gv.dd.gGrid_demand_seventhPartition[(*i)->layer->number][x][y] +=adjGridDemand;
				// }else if(partitionNumber==8){
				// 	gv.dd.gGrid_demand_eighthPartition[(*i)->layer->number][x][y] +=adjGridDemand;
				// }

			}

			adjGridDemand = 0;
		}
	}
}

void movedCellExtraDemand(Inst *Cell, unsigned row, unsigned col)
{
	unsigned cellrow = Cell->row, cellcol = Cell->col;
	//    unsigned rowpos,colpos;
	//
	//    rowpos = row;
	//    colpos = col;

	Cell->row = row;
	Cell->col = col;

	cellExtraDemand(row, col, true);
	if (col < gv.colEndIdx)
		cellExtraDemand(row, col + 1, true);
	if (col > 1)
		cellExtraDemand(row, col - 1, true);

	cellExtraDemand(cellrow, cellcol, true);
	if (cellcol < gv.colEndIdx)
		cellExtraDemand(cellrow, cellcol + 1, true);
	if (cellcol > 1)
		cellExtraDemand(cellrow, cellcol - 1, true);

	Cell->row = cellrow;
	Cell->col = cellcol;

	//    rowpos = cellrow;
	//    colpos = cellcol;
	cellExtraDemand(row, col, false);
	if (col < gv.colEndIdx)
		cellExtraDemand(row, col + 1, false);
	if (col > 1)
		cellExtraDemand(row, col - 1, false);

	cellExtraDemand(cellrow, cellcol, false);
	if (cellcol < gv.colEndIdx)
		cellExtraDemand(cellrow, cellcol + 1, false);
	if (cellcol > 1)
		cellExtraDemand(cellrow, cellcol - 1, false);
}

bool findRoutingForCell(Inst *Cell, vector<Net *>::iterator &itr,
						unsigned rowBegIdx, unsigned rowEndIdx, unsigned colBegIdx,
						unsigned colEndIdx, ofstream &logs)
{
	bool routingFeasible = false;
	

	Net *prevNet = NULL;
	for (vector<Net *>::iterator n = Cell->netList.begin();
		 n != Cell->netList.end(); ++n)
	{
		if ((*n) != prevNet)
		{
			prevNet = (*n);
			routingFeasible = findRoute(*n, rowBegIdx, rowEndIdx, colBegIdx,
										colEndIdx, logs);
			routerCalled++;
			if (routingFeasible == false)
			{
				itr = n;
				//itr++;
				break;
			}
			else
			{
				routerSuccessForAGivenNet++;
			}
		}
	}
	

	return routingFeasible;
}

unsigned commitRouteForCell(Inst *Cell, bool commit, vector<Net *>::iterator itr,
							ofstream &logs)
{
	//if (placerLog)
	//	logs.close();
	//if (routerLog)
	//	logs.open("logs.txt", ios_base::app);
	unsigned totalwl = 0, netwl = 0;
	Net *prevNet = NULL; //initialization important to avoid errors in valgrind
	for (vector<Net *>::iterator n = Cell->netList.begin();
		 n != itr /*Cell->netList.end()*/; ++n)
	{
		if ((*n) != prevNet)
		{
			prevNet = (*n);
			netwl = commitRoute(*n, commit, logs);
			totalwl += netwl;
		}
	}

	//if (routerLog)
	//	logs.close();
	//if (placerLog)
	//	logs.open("logs.txt", ios_base::app);
	return totalwl;
}

unsigned getBoundingBoxCost(routeGuide *source, routeGuide *destination)
{
	unsigned x =
		((source->row) > (destination->row)) ? ((source->row) - (destination->row)) : ((destination->row) - (source->row));
	unsigned y =
		((source->col) > (destination->col)) ? ((source->col) - (destination->col)) : ((destination->col) - (source->col));
	unsigned z =
		((source->lay) > (destination->lay)) ? ((source->lay) - (destination->lay)) : ((destination->lay) - (source->lay));
	return (x + y + z);
}

unsigned pMoveCell(Inst *Cell, unsigned rowBegIdx, unsigned rowEndIdx,
				   unsigned colBegIdx, unsigned colEndIdx,
				   unsigned mCellGridRow, unsigned mCellGridCol, ofstream &logs,unsigned &numCellsMoved) // We should change boundary limitd here (since we don't want boundaries to interfere)
{
	unsigned initial_row, initial_col, temppos; // Initial position o the cell
	//unsigned static numcellsmoved = 0;
	unsigned initial_hpwl, hpwl;
	unsigned initial_wirelength, wirelength;

	int deltahpwl; //delta wirelength
	unsigned newhpwl;

	bool routingFeasible = false;
	int movedflag = 0;
	int overflow = 0;
	unsigned noImprovementIterationCounter = 0;

	//vector<Inst *>::iterator itr;
	vector<Net *>::iterator nItr; // NEt iterator

	overflow = 0;
	initial_hpwl = hpwlForCell(Cell); //  We are claculating HPWL and wirelength seprately
	initial_wirelength = wirelengthForCell(Cell, logs);
	logs << (Cell)->name << " initial HPWL is: " << initial_hpwl
		 << endl;

	if ((Cell)->movable && initial_hpwl != 0)
	{ // Cell is movable then do this
		initial_row = (Cell)->row;
		initial_col = (Cell)->col;

		logs << (Cell)->name << " present at " << (Cell)->row << " "
			 << (Cell)->col << " is movable" << endl;

		if (mCellGridRow >= rowBegIdx && mCellGridCol >= colBegIdx && mCellGridRow <= rowEndIdx && mCellGridCol <= colEndIdx)
		{

			/* deltahpwl = deltahpwlForCell(Cell, mCellGridRow, mCellGridCol, newhpwl, logs);

			if (newhpwl == 0)
			{
				Cell->row = mCellGridRow;
				Cell->col = mCellGridCol;
				movedflag = 2;
			}
			if (deltahpwl < 0)
			{
				Cell->row = mCellGridRow;
				Cell->col = mCellGridCol;
				movedflag = 1;
			}*/
			//accepting without hpwl check
			Cell->row = mCellGridRow; // WE have simply moved if it was in the range
			Cell->col = mCellGridCol;
			movedflag = 1;
		}
		else
		{
			numPredictionFailiures++;
			logs << "mCellGridRow or mCellGridCol or both out of GridLimits" << endl;
		}
		if (movedflag) // Either the cell will be moved or not moved. HEnce if moved then movedFlag will be high here and urther decisions will be taken.
		{
			logs << (Cell)->name << " moved to " << (Cell)->row << " "
				 << (Cell)->col << endl;
		}
		if (initial_row != (Cell)->row || initial_col != (Cell)->col) // This will alwasy be true if any real movement has happened
		{															  //if new position of cell is same as the original position, then nothing to be done
			hpwl = hpwlForCell(Cell);
			//
			if (1) //hpwl < initial_hpwl)	//accepting without hpwl check
			{
				//getDemand();
				//bool blockageOverflow = false;
				movedCellBlockDemand((Cell), initial_row, initial_col);
				movedCellExtraDemand((Cell), initial_row, initial_col);
				/*for (vector<Blockage>::iterator j = Cell->master->blockageList.begin(); j != Cell->master->blockageList.end(); ++j)
				{
					debugDemandFile << gv.dd.gGrid_demand[(*j).layer][Cell->row][Cell->col]  << " at " <<  Cell->row << " " << Cell->col << " " << (*j).layer <<
					", Supply here : " << gv.dd.gGrid_supply[(*j).layer][Cell->row][Cell->col] << endl;
					if(gv.dd.gGrid_demand[(*j).layer][Cell->row][Cell->col] > gv.dd.gGrid_supply[(*j).layer][Cell->row][Cell->col])
					//skipRoutingTrial = true;
					//blockageOverflow = true;
					
				}*/

				routingFeasible = findRoutingForCell(Cell, nItr, rowBegIdx,
													 rowEndIdx, colBegIdx, colEndIdx, logs);
				if (routingFeasible == false)
				{
					routerFailureForAGivenCell++;
					numRouterFailiures++;
					logs << "Routing not feasible for " << (Cell)->name
						 << " at " << (Cell)->row << ", " << (Cell)->col
						 << endl;
					//cellMovedfile << "Routing not feasible for " << (Cell)->name
					//			  << " at " << (Cell)->row << ", " << (Cell)->col
					//			  << endl;
					//algodetail << "Routing not feasible for " << (Cell)->name
					//		   << " at " << (Cell)->row << ", " << (Cell)->col
					//		   << endl;

					commitRouteForCell(Cell, false, nItr, logs);
					temppos = (Cell)->row;
					(Cell)->row = initial_row;
					initial_row = temppos;
					temppos = (Cell)->col;
					(Cell)->col = initial_col;
					initial_col = temppos;
					//getDemand();
					movedCellBlockDemand((Cell), initial_row, initial_col);
					movedCellExtraDemand((Cell), initial_row, initial_col);

					logs << "Movement of " << (Cell)->name
						 << " taken back to " << (Cell)->row << " "
						 << (Cell)->col << endl;
					//algodetail << "Movement of " << (Cell)->name
					//		   << " taken back to " << (Cell)->row << " "
					//		   << (Cell)->col << endl;
				}
				else
				{
					routerSuccessForAGivenCell++;
					numRouterSuccess++;
					wirelength = wirelengthForCell(Cell, logs);
					/*for (unsigned lay = 1; lay <= gv.numLayers && overflow == 0; lay++)
						for (unsigned row = 1; row <= gv.rowEndIdx && overflow == 0; row++)
							for (unsigned col = 1; col <= gv.colEndIdx && overflow == 0; col++)
							{
								overflow += getOverFlow(row, col, lay);
								if (overflow > 0)
								{
									logs << "Overflow at (row,col,lay): " << row << " " << col
										 << " " << lay << endl;
									break; //???????????????????????
								}
							}*/

					for (unsigned lay = 1; lay < gv.numLayers && overflow == 0;
						 lay++)
					//layer traversal necessary, as do not know which all layers are affected by the cell movement
					{
						overflow = getOverFlow(Cell->row, Cell->col, lay);
						overflow +=
							((Cell->col) < gv.colEndIdx) ? getOverFlow(Cell->row, (Cell->col) + 1,
																	   lay)
														 : 0;
						overflow +=
							((Cell->col) > 1) ? getOverFlow(Cell->row, (Cell->col) - 1,
															lay)
											  : 0;
						//not checking the original location, as cell movement would not cause overflow in its original position

						if (overflow > 0)
						{
							//logs << "Overflow at (row,col,lay): "<< row << " " << col << " "<< lay << endl;
							logs << "Overflow after cell moved to row,col while checking for cell moved to: "
								 << Cell->row << " " << Cell->col << " "
								 << "in " << lay << endl;
						}
					}

					if (overflow != 0 /*|| blockageOverflow*/)
					{
						numRouterSuccess--;
						numRouteOverflowFailiures++;
						logs << "Overflow Occured, so movement being taken back"
							 << endl;
						//Delete the routing found from the segment list of the Net
						commitRouteForCell(Cell, false, (Cell)->netList.end(),
										   logs);
						temppos = (Cell)->row;
						(Cell)->row = initial_row;
						initial_row = temppos;
						temppos = (Cell)->col;
						(Cell)->col = initial_col;
						initial_col = temppos;
						//getDemand();
						movedCellBlockDemand((Cell), initial_row, initial_col);
						movedCellExtraDemand((Cell), initial_row, initial_col);
						logs << "Movement of " << (Cell)->name
							 << " taken back to " << (Cell)->row << " "
							 << (Cell)->col << endl;
						//algodetail << "Movement of " << (Cell)->name
						//		   << " taken back to " << (Cell)->row << " "
						//		   << (Cell)->col << endl;
					}
					else
					{

						if (wirelength < initial_wirelength)
						{
							//getDemand();
							commitRouteForCell(Cell, true,
											   (Cell)->netList.end(), logs);
							numCellsMoved++;

							// TEMPORARY STARTS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
							/*for (int i = 0; i < 1248; ++i)
							{
								//X[i]=i;
								//if( i <= 7 || ( i>14 && i<18) || i>24)
								//printf("%f,", X[i]);
								if (i == PYTHON_INPUT_VECTOR_MCELL_INDEX_INDEX_VAL)
								{
									unsigned connectivityRegister = 0x01;

									unsigned itr;
									for (itr = 1; itr <= gv.k; itr++)
									{
										connectivityRegister = 0x01;
										unsigned cellIndex = gv.cellIdxList[itr]; // Picked a cell index (not cell count) -- > But this is correct.
										for (unsigned j = 1; j <= gv.k; j++)
										{
											unsigned connectedCellIndex = gv.cellIdxList[j]; // all the others being considered as connected cell
											connectivityRegister = connectivityRegister << 0x01;
											connectivityRegister |= (gv.connectivity_matrix[cellIndex][connectedCellIndex]) ? 1 : 0;
										}

										predictionLogs << connectivityRegister << ",";
									}
									for (; itr <= NUM_CELLS; itr++)
									{
										predictionLogs << 0 << ",";
									}
								}
								//std::cout << X[i] << ",";
								predictionLogs << X[i] << ",";
								;
								//pValue2 = PyLong_FromLong(X[i]);
								// pValue reference stolen here:
								//PyList_SetItem(pModelPredictFuncArgsObj, i, pValue2);
							}
							predictionLogs << round(Y[0]) << "," << round(Y[1]) << endl;*/
							// TEMPORARY ENDS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
							CellMove *movedCell = new CellMove((Cell),
															   (Cell)->row, (Cell)->col);
							cmListmutex.lock();
							gv.dd.cmList.push_back(movedCell);
							cmListmutex.unlock();
							logs << "Final: " << (Cell)->name
								 << " moved to " << (Cell)->row << " "
								 << (Cell)->col << endl;
						REMOVETHIS:
							if ((Cell)->name == "C336")
								//(-1);
								//algodetail << "Final: " << (Cell)->name
								//		   << " moved to " << (Cell)->row << " "
								//		   << (Cell)->col << endl;

								logs << (Cell)->name << " initial HPWL was: " << initial_hpwl << endl;
							//algodetail << (Cell)->name << " initial HPWL was: "
							//		   << initial_hpwl << endl;

							logs << "Final " << (Cell)->name
								 << " HPWL is: " << hpwl << endl;
							//algodetail << "Final " << (Cell)->name
							//		   << " HPWL is: " << hpwl << endl;
						}
						else
						{
							logs
								<< "New HPWL is lesser but new wirelength is greater"
								<< endl;
							numPredictionFailiuresWirelength++;
							//Delete the routing found from the segment list of the Net
							commitRouteForCell(Cell, false, (Cell)->netList.end(), logs);

							temppos = (Cell)->row;
							(Cell)->row = initial_row;
							initial_row = temppos;
							temppos = (Cell)->col;
							(Cell)->col = initial_col;
							initial_col = temppos;
							//getDemand();
							movedCellBlockDemand((Cell), initial_row, initial_col);
							movedCellExtraDemand((Cell), initial_row, initial_col);

							logs << "Movement of " << (Cell)->name
								 << " taken back to " << (Cell)->row << " "
								 << (Cell)->col << endl;
							//algodetail << "Movement of " << (Cell)->name
							//		   << " taken back to " << (Cell)->row << " "
							//		   << (Cell)->col << endl;
						}
					}
				}
			}
			else
			{ //if hpwl >=initial_hpwl, then movement needs to be taken back as of no use to move unnecessarily
				logs << "Final: " << (Cell)->name << " moved to "
					 << (Cell)->row << " " << (Cell)->col << endl;
				//algodetail << "Final: " << (Cell)->name << " moved to "
				//						   << (Cell)->row << " " << (Cell)->col << endl;

				logs << (Cell)->name << " initial HPWL was: "
					 << initial_hpwl << endl;
				//algodetail << (Cell)->name << " initial HPWL was: "
				//		   << initial_hpwl << endl;

				logs << "After movement " << (Cell)->name
					 << " HPWL is: " << hpwl << endl;
				//algodetail << "After movement " << (Cell)->name << " HPWL is: "
				//		   << hpwl << endl;

				(Cell)->row = initial_row;
				(Cell)->col = initial_col;

				logs << "Movement of " << (Cell)->name
					 << " taken back to " << (Cell)->row << " "
					 << (Cell)->col << endl;
				//algodetail << "Movement of " << (Cell)->name
				//		   << " taken back to " << (Cell)->row << " "
				//		   << (Cell)->col << endl;
			}
		}
		else
		{
			numNotFoundBetterSolution++;
			logs << "Algo could not find a better cell position" << endl;
		}
	}
	else if (initial_hpwl == 0)
	{
		initialHPWLZero++;
		logs << (Cell)->name << " present at " << (Cell)->row << " "
			 << (Cell)->col << " need NOT be moved" << endl;
		//cellMovedfile << (Cell)->name << " present at " << (Cell)->row << " "
		//			  << (Cell)->col << " need NOT be moved" << endl;
		//algodetail << (Cell)->name << " present at " << (Cell)->row << " "
		//		   << (Cell)->col << " need NOT be moved " << endl;
	}
	else
	{
		logs << (Cell)->name << " present at " << (Cell)->row << " "
			 << (Cell)->col << " is NOT movable" << endl;
		//cellMovedfile << (Cell)->name << " present at " << (Cell)->row << " "
		//			  << (Cell)->col << " is NOT movable" << endl;
		//algodetail << (Cell)->name << " present at " << (Cell)->row << " "
		//		   << (Cell)->col << " is NOT movable" << endl;
	}
	predictionLogs << movedflag << "," << endl;
	//	}
	logs << "Max Cell allowed to move are: " << gv.maxCellMove << endl;
	logs << "Number of Cells Moved = " << numCellsMoved << endl;

	return numCellsMoved;
}

void initializeRoute(unsigned rowStart, unsigned rowEnd, unsigned colStart,
					 unsigned colEnd)
{

	unsigned row, col, lay;
	unsigned initialWirelengthCost = 0, initialPathCost = 0;

	//	routeGuide *cell;
	//cout<<" In initialize Route" <<endl;
	for (lay = 1; lay <= gv.numLayers; lay++)
	{
		initialWirelengthCost = 0;
		initialPathCost = 0;
		for (row = rowStart; row <= rowEnd; row++)
		{
			for (col = colStart; col <= colEnd; col++)
			{
				(gv.dd.gGrid_Route[lay][row][col])->row = row;
				(gv.dd.gGrid_Route[lay][row][col])->col = col;
				(gv.dd.gGrid_Route[lay][row][col])->lay = lay;
				(gv.dd.gGrid_Route[lay][row][col])->pathCost = initialPathCost;
				(gv.dd.gGrid_Route[lay][row][col])->wireLengthCost = initialWirelengthCost;
				(gv.dd.gGrid_Route[lay][row][col])->pred = 'X';
				(gv.dd.gGrid_Route[lay][row][col])->reached = false;
				(gv.dd.gGrid_Route[lay][row][col])->expanded = false;

				/*cout<< "row: "<<(gv.dd.gGrid_Route[lay][row][col])->row<<" col: "<<(gv.dd.gGrid_Route[lay][row][col])->col
						<<" lay: "<<(gv.dd.gGrid_Route[lay][row][col])->lay << endl;*/
			}
		}
	}
}

void initializeRouteGrid(unsigned rowStart, unsigned rowEnd, unsigned colStart,
						 unsigned colEnd)
{
	unsigned row, col, lay;
	unsigned initialWirelengthCost = 0, initialPathCost = 0;

	routeGuide *cell;
	//cout<<" In initialize Route Grid" <<endl;
	for (lay = 1; lay <= gv.numLayers; lay++)
	{
		initialWirelengthCost = 0;
		initialPathCost = 0;
		for (row = rowStart; row <= rowEnd; row++)
		{
			for (col = colStart; col <= colEnd; col++)
			{
				cell = new routeGuide(row, col, lay, initialPathCost, initialWirelengthCost, 'X', false, false);
				gv.dd.gGrid_Route[lay][row][col] = cell;
				/*cout<< "row: "<<(gv.dd.gGrid_Route[lay][row][col])->row<<" col: "<<(gv.dd.gGrid_Route[lay][row][col])->col
						<<" lay: "<<(gv.dd.gGrid_Route[lay][row][col])->lay << endl;*/
			}
		}
	}
}

int backtrack(routeGuide *cell, Net *N, ofstream &logs)
{

	routeGuide *currCell = cell;
	unsigned startSegment, endSegment;
	direction dir;
	char prev = 'A';
	char curr = 'A';
	unsigned row = cell->row;
	unsigned col = cell->col;
	unsigned lay = cell->lay;
	unsigned rowStart = 0, colStart = 0, layStart = 0, rowEnd = 0, colEnd = 0, layEnd = 0;
	while (gv.dd.gGrid_Route[lay][row][col]->pred != 'X')
	{

		logs << row << " " << col << " " << lay << " "
			 << gv.dd.gGrid_Route[lay][row][col]->pred << endl;

		curr = gv.dd.gGrid_Route[lay][row][col]->pred;

		if (prev == 'A')
		{
			rowStart = row;
			colStart = col;
			layStart = lay;
			rowEnd = row;
			colEnd = col;
			layEnd = lay;

			if (curr == 'U')
			{
				lay = lay + 1;
				layEnd = lay;
			}
			else if (curr == 'D')
			{
				lay = lay - 1;
				layEnd = lay;
			}
			else if (curr == 'N')
			{
				row = row + 1;
				rowEnd = row;
			}
			else if (curr == 'S')
			{
				row = row - 1;
				rowEnd = row;
			}
			else if (curr == 'E')
			{
				col = col + 1;
				colEnd = col;
			}
			else if (curr == 'W')
			{
				col = col - 1;
				colEnd = col;
			}

			prev = curr;
		}
		else if (prev == curr)
		{
			if (curr == 'U')
			{
				lay = lay + 1;
				layEnd = lay;
			}
			else if (curr == 'D')
			{
				lay = lay - 1;
				layEnd = lay;
			}
			else if (curr == 'N')
			{
				row = row + 1;
				rowEnd = row;
			}
			else if (curr == 'S')
			{
				row = row - 1;
				rowEnd = row;
			}
			else if (curr == 'E')
			{
				col = col + 1;
				colEnd = col;
			}
			else if (curr == 'W')
			{
				col = col - 1;
				colEnd = col;
			}

			prev = curr;
		}
		else
		{
			if (rowStart != rowEnd)
			{
				dir = ALONG_ROW;
			}
			else if (colStart != colEnd)
			{
				dir = ALONG_COL;
			}
			else if (layStart != layEnd)
			{
				dir = ALONG_Z;
			}
			Route *segRoute = new Route(rowStart, colStart, layStart, rowEnd,
										colEnd, layEnd, dir, N);
			logs << rowStart << " " << colStart << " " << layStart << " "
				 << rowEnd << " " << colEnd << " " << layEnd << " "
				 << N->name << endl;
			N->segmentList.push_back(segRoute);

			rowStart = rowEnd;
			colStart = colEnd;
			layStart = layEnd;
			if (curr == 'U')
			{
				lay = lay + 1;
				layEnd = lay;
			}
			else if (curr == 'D')
			{
				lay = lay - 1;
				layEnd = lay;
			}
			else if (curr == 'N')
			{
				row = row + 1;
				rowEnd = row;
			}
			else if (curr == 'S')
			{
				row = row - 1;
				rowEnd = row;
			}
			else if (curr == 'E')
			{
				col = col + 1;
				colEnd = col;
			}
			else if (curr == 'W')
			{
				col = col - 1;
				colEnd = col;
			}
			prev = curr;
		}
	}
	if (rowStart != rowEnd)
	{
		dir = ALONG_ROW;
	}
	else if (colStart != colEnd)
	{
		dir = ALONG_COL;
	}
	else if (layStart != layEnd)
	{
		dir = ALONG_Z;
	}
	Route *segRoute = new Route(rowStart, colStart, layStart, rowEnd, colEnd,
								layEnd, dir, N);
	logs << rowStart << " " << colStart << " " << layStart << " " << rowEnd
		 << " " << colEnd << " " << layEnd << " " << N->name << endl;
	N->segmentList.push_back(segRoute);

	return 1;
}

int isEqual(routeGuide *x, routeGuide *y)
{

	if ((x->row == y->row) && (x->col == y->col) && (x->lay == y->lay))
		return 1;
	else
		return 0;
}

bool getNbforGivenCell(unsigned rowStart, unsigned rowEnd, unsigned colStart,
					   unsigned colEnd, routeGuide *cell, routeGuide *destination, Net *N,
					   unsigned minLayer,
					   priority_queue<routeGuide *, vector<routeGuide *>, CompareCellCost> &tempWf,
					   unsigned &wirelengthCost, ofstream &logs)
{
	//	vector<routeGuide *> nb;
	bool destinationFound = false;
	unsigned row = cell->row;
	unsigned col = cell->col;
	unsigned lay = cell->lay;
	routeGuide *cell1;
	routeGuide *cell2;
	routeGuide *cell3;
	routeGuide *cell4;

	wirelengthCost = cell->wireLengthCost + 1;

	logs << "Row start: " << rowStart << " Row end: " << rowEnd << " Col start: " << colStart << " Col end: " << colEnd
		 << "Min Layer: " << minLayer << endl;

	logs << "Cell row " << row << " col " << col << "lay " << lay << endl;

	if (lay % 2 != 0)
	{ //Horizontal movement only
		if ((col + 1) <= colEnd && gv.dd.gGrid_Route[lay][row][col + 1]->expanded != true)
		{
			float cost = (float)(getCongestion(row, col + 1, lay) * 100.0);
			//logs << "row: "<< row << "col: "<< col + 1 << "lay: "<< lay << " congestion: "<< cost << endl;

			if (cost < 100.0)
			{
				cell1 = gv.dd.gGrid_Route[lay][row][col + 1];
				/*if(cell1 == NULL)
					logs << " row: " << row << " col: " << col +1 << " lay: "
					 << lay << "cell is NULL"<< endl;
				*/
				cell1->reached = true;
				cell1->expanded = false;
				cell1->wireLengthCost = wirelengthCost;
				//	cell1 = new routeGuide(row, col + 1, lay, wirelengthCost, 'W',
				//						   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell1,
															  destination);
				cell1->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay][row][col + 1]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay][row][col + 1]->pred = 'W';
				//logs << " row: " << row << " col: " << col + 1 << " lay: "<< lay << "in getNb"<< endl;
				if (isEqual(cell1, destination))
				{
					destinationFound = true;
					backtrack(cell1, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
					//logs << "dest found : " << getCongestion(7, 4, 1) << endl;
				}

				//.push_back(cell1);
				tempWf.push(cell1);
				//logs << "If block over"<<endl;
			}
		}
		if ((col - 1) >= colStart && gv.dd.gGrid_Route[lay][row][col - 1]->expanded != true)
		{
			/*if (row == 7 && col - 1 == 4 && lay == 1) {
			 logs << "found" << endl;
			 }*/
			float cost = (float)(getCongestion(row, col - 1, lay) * 100.0);
			// << "row: "<< row << "col: "<< col - 1 << "lay: "<< lay << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{
				cell2 = gv.dd.gGrid_Route[lay][row][col - 1];
				//if(cell2 == NULL)
				// << " row: " << row << " col: " << col -1 << " lay: "
				// << lay << "cell is NULL"<< endl;
				cell2->reached = true;
				cell2->expanded = false;
				cell2->wireLengthCost = wirelengthCost;
				//cell2 = new routeGuide(row, col - 1, lay, wirelengthCost, 'E',
				//				   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell2,
															  destination);
				cell2->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay][row][col - 1]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay][row][col - 1]->pred = 'E';
				// << " row: " << row << " col: " << col - 1 << " lay: "<< lay << "in getNb"<< endl;
				if (isEqual(cell2, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell2, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/

					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
					//logs << "dest found : " << getCongestion(7, 4, 1) << endl;
				}
				//.push_back(cell2);
				tempWf.push(cell2);
				// << "If block over"<<endl;
			}
		}
		if ((lay - 1) >= minLayer && gv.dd.gGrid_Route[lay - 1][row][col]->expanded != true)
		{
			/*if (row == 7 && col == 4 && lay - 1 == 1) {
			 logs << "found" << endl;
			 }*/
			float cost = (float)(getCongestion(row, col, lay - 1) * 100.0);
			// << "row: "<< row << "col: "<< col << "lay: "<< lay -1 << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{
				cell3 = gv.dd.gGrid_Route[lay - 1][row][col];
				//if(cell3 == NULL)
				//<< " row: " << row << " col: " << col << " lay: " << lay - 1 << "cell is NULL" << endl;

				cell3->reached = true;
				cell3->expanded = false;
				cell3->wireLengthCost = wirelengthCost;
				// cell3 = new routeGuide(row, col, lay - 1, wirelengthCost, 'U',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell3,
															  destination);
				cell3->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay - 1][row][col]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay - 1][row][col]->pred = 'U'; //Should not this be Down, i.e. D?
				//logs << " row: " << row << " col: " << col << " lay: "<< lay - 1 << "in getNb"<< endl;
				if (isEqual(cell3, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell3, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
					//logs << "dest found : " << getCongestion(7, 4, 1) << endl;
				}
				//.push_back(cell3);
				tempWf.push(cell3);
				//logs << "If block over"<<endl;
			}
		}
		if ((lay + 1) <= gv.numLayers && gv.dd.gGrid_Route[lay + 1][row][col]->expanded != true)
		{
			/*if (row == 7 && col == 4 && lay + 1 == 1) {
			 logs << "found" << endl;
			 }*/
			float cost = (float)(getCongestion(row, col, lay + 1) * 100.0);
			//logs << "row: "<< row << "col: "<< col << "lay: "<< lay + 1 << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{

				cell4 = gv.dd.gGrid_Route[lay + 1][row][col];
				//if(cell4 == NULL)
				//logs<< " row: " << row << " col: " << col << " lay: " << lay + 1 << "cell is NULL" << endl;

				cell4->reached = true;
				cell4->expanded = false;
				cell4->wireLengthCost = wirelengthCost;
				// cell4 = new routeGuide(row, col, lay + 1, wirelengthCost, 'D',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell4,
															  destination);
				cell4->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay + 1][row][col]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay + 1][row][col]->pred = 'D'; //Should not this be Up, i.e. U?
				//logs << " row: " << row << " col: " << col << " lay: "
				//<< lay + 1 << "in getNb"<< endl;
				if (isEqual(cell4, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell4, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
					//logs << "dest found : " << getCongestion(7, 4, 1) << endl;
				}
				//.push_back(cell4);
				tempWf.push(cell4);
				//logs << "If block over"<<endl;
			}
		}
	}
	else
	{ //Ve
		if ((row + 1) <= rowEnd && gv.dd.gGrid_Route[lay][row + 1][col]->expanded != true)
		{
			float cost = (float)(getCongestion(row + 1, col, lay) * 100.0);
			//logs << "row: "<< row +1 << "col: "<< col << "lay: "<< lay << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{

				cell1 = gv.dd.gGrid_Route[lay][row + 1][col];
				//if(cell1 == NULL)
				//logs << " row: " << row + 1 << " col: " << col << " lay: "
				//<< lay << "cell is NULL"<< endl;
				cell1->reached = true;
				cell1->expanded = false;
				cell1->wireLengthCost = wirelengthCost;
				// cell1 = new routeGuide(row + 1, col, lay, wirelengthCost, 'S',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell1,
															  destination);
				cell1->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay][row + 1][col]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay][row + 1][col]->pred = 'S'; //Wouldn't this be North, taking origin to be at bottom left corner
				//logs << " row: " << row + 1 << " col: " << col << " lay: "
				//<< lay << "in getNb"<< endl;
				if (isEqual(cell1, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell1, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
				}
				//.push_back(cell1);
				tempWf.push(cell1);
				//logs << "If block over"<<endl;
			}
		}
		if ((row - 1) >= rowStart && gv.dd.gGrid_Route[lay][row - 1][col]->expanded != true)
		{
			float cost = (float)(getCongestion(row - 1, col, lay) * 100.0);
			//logs << "row: "<< row - 1 << "col: "<< col << "lay: "<< lay << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{

				cell2 = gv.dd.gGrid_Route[lay][row - 1][col];
				cell2->reached = true;
				cell2->expanded = false;
				cell2->wireLengthCost = wirelengthCost;
				// cell2 = new routeGuide(row - 1, col, lay, wirelengthCost, 'N',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell2,
															  destination);
				cell2->pathCost = boundingBoxCost;
				//if(cell2 == NULL)
				//logs<< " row: " << row - 1 << " col: " << col << " lay: " << lay + 1 << "cell is NULL" << endl;
				gv.dd.gGrid_Route[lay][row - 1][col]->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay][row - 1][col]->pred = 'N'; //South??
				//logs << " row: " << row - 1 << " col: " << col << " lay: "
				// << lay << "in getNb"<< endl;
				if (isEqual(cell2, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell2, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
				}
				//.push_back(cell2);
				tempWf.push(cell2);
				//logs << "If block over"<<endl;
			}
		}
		if ((lay - 1) >= minLayer && gv.dd.gGrid_Route[lay - 1][row][col]->expanded != true)
		{
			/*if (row == 7 && col == 4 && lay - 1 == 1) {
			 logs << "found" << endl;
			 }*/
			float cost = (float)(getCongestion(row, col, lay - 1) * 100.0);
			//logs << "row: "<< row << "col: "<< col << "lay: "<< lay -1 << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{

				cell3 = gv.dd.gGrid_Route[lay - 1][row][col];

				//if(cell3 == NULL)
				//logs<< " row: " << row << " col: " << col << " lay: " << lay - 1 << "cell is NULL" << endl;
				cell3->reached = true;
				cell3->expanded = false;
				cell3->wireLengthCost = wirelengthCost;
				// cell3 = new routeGuide(row, col, lay - 1, wirelengthCost, 'U',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell3,
															  destination);
				cell3->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay - 1][row][col]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay - 1][row][col]->pred = 'U';
				//logs << " row: " << row << " col: " << col << " lay: "
				//<< lay - 1 << "in getNb"<< endl;
				if (isEqual(cell3, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell3, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
					//logs << "dest found" << getCongestion(7, 4, 1) << endl;
				}
				//.push_back(cell3);
				tempWf.push(cell3);
				//logs << "If block over"<<endl;
			}
		}
		if ((lay + 1) <= gv.numLayers && gv.dd.gGrid_Route[lay + 1][row][col]->expanded != true)
		{
			float cost = (float)(getCongestion(row, col, lay + 1) * 100.0);
			//logs << "row: "<< row << "col: "<< col << "lay: "<< lay +1 << " congestion: "<< cost << endl;
			if (cost < 100.0)
			{

				cell4 = gv.dd.gGrid_Route[lay + 1][row][col];

				//if(cell4 == NULL)
				//logs<< " row: " << row << " col: " << col << " lay: " << lay + 1 << "cell is NULL" << endl;
				cell4->reached = true;
				cell4->expanded = false;
				cell4->wireLengthCost = wirelengthCost;
				// cell4 = new routeGuide(row, col, lay + 1, wirelengthCost, 'D',
				// 					   true, false);
				unsigned boundingBoxCost = getBoundingBoxCost(cell4,
															  destination);
				cell4->pathCost = boundingBoxCost;
				gv.dd.gGrid_Route[lay + 1][row][col]->pathCost =
					boundingBoxCost;
				gv.dd.gGrid_Route[lay + 1][row][col]->pred = 'D';
				//logs << " row: " << row << " col: " << col << " lay: "
				//<< lay + 1 << "in getNb"<< endl;
				if (isEqual(cell4, destination))
				{

					//	logs << "target reached" << endl;
					destinationFound = true;
					//netWirelengthDemand(N, true, true);
					backtrack(cell4, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					return destinationFound;
					/*while (!wavefront.empty())
					 wavefront.pop();*/
					//netWirelengthDemand(N, true, false); //Calculating and pdating the segment demands of the routed net in the 3D array
				}
				//.push_back(cell4);
				tempWf.push(cell4);
				//logs << "If block over"<<endl;
			}
		}
	}
	//	logs << "GetNb Over with destinationFound = " << destinationFound <<endl;
	return destinationFound;
}

bool findRoute(Net *N, unsigned rowStart, unsigned rowEnd, unsigned colStart,
			   unsigned colEnd, ofstream &logs)
{
	routeGuide *source;
	routeGuide *destination;
	unsigned wirelengthCost;
	bool routeSuccess = true;
	bool segmentPresent = false;
	bool noRouteSegment; //to check for nets with all pins in same gGrid and NO min layer constraint
	bool allPinsInSameLayer = false;
	vector<Route *>::iterator itr1, itr2;
	unsigned minLayer = 1;
	bool minLayerCstr = false;
	//logs << "ov1 : " << getCongestion(4, 1, 1) << endl;
	netWirelengthDemand(N, true, true); //Decrease the segment demands of the current net
	//logs << "ov2 : " << getCongestion(4, 1, 1) << endl;
	if (!N->segmentList.empty())
		segmentPresent = true;
	N->segmentList.clear(); //clearing segment list of the Net to be routed
	//if (segmentPresent)
	//if (routerLog)
	//	logs.close();
	//if (steinerPointsLog)
	//	logs.open("logs.txt", ios_base::app);
	//allPinsInSameLayer = createRouterPinlist(N, logs);
	//createMultiLayerRouterPinlist(N, logs);
	//if (steinerPointsLog)
	//	logs.close();
	//if (routerLog)
	//	logs.open("logs.txt", ios_base::app);
	//	getDemand();

	initializeRoute(rowStart, rowEnd, colStart, colEnd);
	allPinsInSameLayer = false; //to NEVER run stenier: issues in FLUTE with python
	if (allPinsInSameLayer)
	{

		for (vector<pinsForRouting *>::iterator i = N->routerPinList.begin();
			 i + 1 < N->routerPinList.end() && routeSuccess;)
		{ //i++ being done inside the loop after fetching the source pin details

			priority_queue<routeGuide *, vector<routeGuide *>, CompareWirelengthCost> wavefront;
			priority_queue<routeGuide *, vector<routeGuide *>, CompareCellCost> tempWf;
			routeSuccess = false;
			bool sourceViaSuccess = false;
			bool destViaSuccess = false;
			Route *sourceViaSegRoute;
			Route *destViaSegRoute;
			//source = new routeGuide((*i)->row, (*i)->col, (*i)->layer, 0, 0, 'X',
			//						true, false);
			source = gv.dd.gGrid_Route[(*i)->layer][(*i)->row][(*i)->col];
			source->pathCost = 0;
			source->wireLengthCost = 0;
			source->pred = 'X';
			source->reached = true;
			source->expanded = false;
			logs << "source : "
				 << "NET: " << N->name << " row: " << source->row
				 << " col: " << source->col << " lay: " << source->lay
				 << endl;
			i++;
			// destination = new routeGuide((*i)->row, (*i)->col, (*i)->layer, 0, 0,
			// 							 'X', true, false);
			destination = gv.dd.gGrid_Route[(*i)->layer][(*i)->row][(*i)->col];
			destination->pathCost = 0;
			destination->wireLengthCost = 0;
			destination->pred = 'X';
			destination->reached = true;
			destination->expanded = false;
			logs << "destination : "
				 << "NET: " << N->name << " row: "
				 << destination->row << " col: " << destination->col
				 << " lay: " << destination->lay << endl;
			//logs << "during : " << getCongestion(3, 3, 1) << endl;
			bool pinsInsameGrid = false;
			if (isEqual(source, destination))
				pinsInsameGrid = true;

			if (N->layer)
			{
				minLayer = N->layer->number;
				minLayerCstr = true;
			}
			else
			{
				minLayer = 1;
				minLayerCstr = false;
			}

			if (pinsInsameGrid == true) //minLayerCstr == true)// && minLayer == source->lay && segmentPresent == true)
			{
				if (minLayerCstr == true)
				{
					if (minLayer == source->lay)
					{
						if (segmentPresent == true)
						{
							//netWirelengthDemand(N, true, false);
							noRouteSegment = true;
							routeSuccess = true;
							//if (allPinsInSameLayer)
							//	i++;
							//continue;
						}
						else //if segmentPresent == false
						{
							//netWirelengthDemand(N, true, false);
							noRouteSegment = true;
							//if (allPinsInSameLayer)
							//	i++;
							//continue; //moves to next iteration of the loop, i.e. the next pin
						}
					}
					else //if (pinsInsameGrid && minLayerCstr == true &&  minLayer != source->lay)
						 //was giving issue without use of minLayerCstr boolean variable, as then minLayer was set to 1
						 //so for conditions where NoCstr for Net, and pin layer != 1, unneccasry via gets created
						 //and was giving overflow ¥¥s
					{
						bool overflowExist = false;
						for (unsigned i = source->lay; i <= minLayer; i++) //if source->lay > minLayer then doesn't execute
						{
							if (getCongestion(source->row, source->col, i) * 100.0 >= 100.0)
							{
								routeSuccess = false;
								noRouteSegment = false;
								overflowExist = true;
								break;
							}
						}

						if (overflowExist)
						{
							logs << "pinsInsameGrid : overflow during via insertion" << endl;
							break;
						}
						//sourceViaSuccess = true;
						routeSuccess = true; //as the via gets pushed into the segment list only if routeSuccess is true
						//Above lines not useful, as the loop continues further without reaching the if(sourceViaSuccess)
						sourceViaSegRoute = new Route(source->row, source->col, source->lay,
													  source->row, source->col, minLayer, ALONG_Z, N);
						N->segmentList.push_back(sourceViaSegRoute);

						/*
						//logs << "via before : " << getCongestion(3, 3, 1) << endl;
						sourceViaSegRoute = new Route(source->row, source->col,
													  source->lay, source->row, source->col, minLayer,
													  ALONG_Z, N);
						//why not overflow check present for the via being inserted here? _PRJ

						N->segmentList.push_back(sourceViaSegRoute);
						//netWirelengthDemand(N, true, false);
						noRouteSegment = false;
						routeSuccess = true;
						//if (allPinsInSameLayer)
							//i++;
						//logs << "via after : " << getCongestion(3, 3, 1) << endl;
						//continue;
*/
					}
				}
				else //if minLayerCstr is false, but pins are in the same grid
				{
					if (segmentPresent == true)
					{
						//netWirelengthDemand(N, true, false);
						noRouteSegment = true;
						routeSuccess = true;
						//if (allPinsInSameLayer)
						//	i++;
						//continue;
					}
					else //if segmentPresent == false
					{
						//netWirelengthDemand(N, true, false);
						noRouteSegment = true;
						routeSuccess = false;
						//if (allPinsInSameLayer)
						//	i++;
						//continue; //moves to next iteration of the loop, i.e. the next pin
					}
				}

				if (allPinsInSameLayer)
					i++;
				//logs << "via after : " << getCongestion(3, 3, 1) << endl;
				continue;
			}
			else
			{
				noRouteSegment = false;
			}

			if (getCongestion(source->row, source->col, source->lay) >= 1)
			{
				routeSuccess = false;
				if (segmentPresent)
					noRouteSegment = false;
				break;
			}

			if (!pinsInsameGrid && source->lay < minLayer) //'<' used as if source->lay > minLayer, then no need to put via
			{
				bool overflowExist = false;
				for (unsigned i = source->lay; i <= minLayer; i++)
				{
					if (getCongestion(source->row, source->col, i) * 100.0 >= 100.0)
					{
						routeSuccess = false;
						noRouteSegment = false;
						overflowExist = true;
						break;
					}
				}

				if (overflowExist)
				{
					logs << "source : overflow during via insertion" << endl;
					break;
				}
				sourceViaSuccess = true;
				sourceViaSegRoute = new Route(source->row, source->col, source->lay,
											  source->row, source->col, minLayer, ALONG_Z, N);

				N->segmentList.push_back(sourceViaSegRoute);
				//source->lay = minLayer;
				source = gv.dd.gGrid_Route[minLayer][source->row][source->col];
			}

			if (!pinsInsameGrid && destination->lay < minLayer)
			{
				bool overflowExist = false;
				for (unsigned i = destination->lay; i <= minLayer; i++)
				{
					if (getCongestion(destination->row, destination->col, i) * 100.0 >= 100.0)
					{
						routeSuccess = false;
						noRouteSegment = false;
						overflowExist = true;
						break;
					}
				}

				if (overflowExist)
				{
					logs << "destination : overflow during via insertion"
						 << endl;
					break;
				}
				destViaSuccess = true;
				destViaSegRoute = new Route(destination->row, destination->col,
											destination->lay, destination->row, destination->col,
											minLayer, ALONG_Z, N);

				N->segmentList.push_back(destViaSegRoute);
				//destination->lay = minLayer;
				destination = gv.dd.gGrid_Route[minLayer][destination->row][destination->col];
			}

			if (isEqual(source, destination))
			{
				routeSuccess = true;
				if (allPinsInSameLayer)
					i++;
				continue;
			}
			if (getCongestion(destination->row, destination->col,
							  destination->lay) >= 1)
			{
				routeSuccess = false;
				if (segmentPresent)
					noRouteSegment = false;
				break;
			}
			wavefront.push(source);
			gv.dd.gGrid_Route[source->lay][source->row][source->col] = source;
			//gv.dd.q.push(source);
			wirelengthCost = 0;
			auto findRouteStart = std::chrono::system_clock::now();
			while (!wavefront.empty())
			{
				auto findRouteFinish = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedFindRoute = (findRouteFinish - findRouteStart);
				double time_taken = elapsedFindRoute.count();
				//logs << ctime(&moveCellFinish) << endl;

				time_taken /= 60;
				if (time_taken >= 60)
				{
					logs << "In Find Route, Duration >= " << time_taken << " min." << endl;
					routeSuccess = false;
					noRouteSegment = false;
					break;
				}
				routeGuide *cell = wavefront.top();
				wavefront.pop();
				//routeGuide *cell = gv.dd.q.front();
				cell->expanded = true;
				gv.dd.gGrid_Route[cell->lay][cell->row][cell->col]->expanded =
					true;
				/*logs << " row: " << cell->row << " col: " << cell->col << " lay: "
				 << cell->lay << " popped" << endl;*/

				if (isEqual(cell, destination))
				{
					logs << "target reached 1" << endl;
					backtrack(cell, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
					routeSuccess = true;
					break;
				}

				//vector<routeGuide *> nb;
				bool destFound = false;
				destFound = getNbforGivenCell(rowStart, rowEnd, colStart, colEnd, cell, destination,
											  N, minLayer, tempWf, wirelengthCost, logs);
				if (destFound)
				{
					logs << "target reached 2" << endl;
					routeSuccess = true;

					while (!tempWf.empty())
					{
						//routeGuide *oldWf = tempWf.top();
						//logs << "Deleting from temporary wave front" << endl;
						//delete oldWf;
						tempWf.pop();
					}
					break;
				}

				while (!tempWf.empty())
				{
					routeGuide *r = tempWf.top();
					if (isEqual(r, destination))
					{
						logs << "found" << endl;
					}
					r->expanded = true;
					/*logs << " row: " << r->row << " col: " << r->col << " lay: "
					 << r->lay << "pred: " << r->pred << " added" << endl;*/
					wavefront.push(r);
					tempWf.pop();
				}
			}
			/*			if (routeSuccess)
			{
				if (sourceViaSuccess)
				{
					//netWirelengthDemand(N, true, true);
					N->segmentList.push_back(sourceViaSegRoute);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
				}
				if (destViaSuccess)
				{
					//netWirelengthDemand(N, true, true);
					N->segmentList.push_back(destViaSegRoute);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
				}
			}*/
			if (allPinsInSameLayer)
				i++;
		}

		logs << "For loop of routerPinList over" << endl;
		for (vector<pinsForRouting *>::iterator i = N->routerPinList.begin(); i != N->routerPinList.end(); i++)
			delete *i;
		N->routerPinList.clear();
	}

	else
	{

		for (vector<Pin *>::iterator i = N->pinList.begin(); i + 1 < N->pinList.end() && routeSuccess;)
		{ //i++ being done inside the loop after fetching the source pin details

			priority_queue<routeGuide *, vector<routeGuide *>, CompareWirelengthCost> wavefront;
			priority_queue<routeGuide *, vector<routeGuide *>, CompareCellCost> tempWf;
			routeSuccess = false;
			bool sourceViaSuccess = false;
			bool destViaSuccess = false;
			Route *sourceViaSegRoute;
			Route *destViaSegRoute;
			//source = new routeGuide((*i)->inst->row, (*i)->inst->col, (*i)->masterPin->layer, 0, 0, 'X',
			// 						true, false);

			source = gv.dd.gGrid_Route[(*i)->masterPin->layer][(*i)->inst->row][(*i)->inst->col];
			source->pathCost = 0;
			source->wireLengthCost = 0;
			source->pred = 'X';
			source->reached = true;
			source->expanded = false;
			logs << "source : "
				 << "NET: " << N->name << " row: " << source->row
				 << " col: " << source->col << " lay: " << source->lay
				 << endl;
			i++;
			//destination = new routeGuide((*i)->row, (*i)->col, (*i)->layer, 0, 0,
			// 							 'X', true, false);
			destination = gv.dd.gGrid_Route[(*i)->masterPin->layer][(*i)->inst->row][(*i)->inst->col];
			destination->pathCost = 0;
			destination->wireLengthCost = 0;
			destination->pred = 'X';
			destination->reached = true;
			destination->expanded = false;
			logs << "destination : "
				 << "NET: " << N->name << " row: "
				 << destination->row << " col: " << destination->col
				 << " lay: " << destination->lay << endl;
			//logs << "during : " << getCongestion(3, 3, 1) << endl;
			bool pinsInsameGrid = false;
			if (isEqual(source, destination))
				pinsInsameGrid = true;

			if (N->layer)
			{
				minLayer = N->layer->number;
				minLayerCstr = true;
			}
			else
			{
				minLayer = 1;
				minLayerCstr = false;
			}

			if (pinsInsameGrid == true) //minLayerCstr == true)// && minLayer == source->lay && segmentPresent == true)
			{
				if (minLayerCstr == true)
				{
					if (minLayer == source->lay)
					{
						if (segmentPresent == true)
						{
							//netWirelengthDemand(N, true, false);
							noRouteSegment = true;
							routeSuccess = true;
							//if (allPinsInSameLayer)
							//	i++;
							//continue;
						}
						else //if segmentPresent == false
						{
							//netWirelengthDemand(N, true, false);
							noRouteSegment = true;
							//if (allPinsInSameLayer)
							//	i++;
							//continue; //moves to next iteration of the loop, i.e. the next pin
						}
					}
					else //if (pinsInsameGrid && minLayerCstr == true &&  minLayer != source->lay)
						 //was giving issue without use of minLayerCstr boolean variable, as then minLayer was set to 1
						 //so for conditions where NoCstr for Net, and pin layer != 1, unneccasry via gets created
						 //and was giving overflow errors
					{
						bool overflowExist = false;
						for (unsigned i = source->lay; i <= minLayer; i++) //if source->lay > minLayer then doesn't execute
						{
							if (getCongestion(source->row, source->col, i) * 100.0 >= 100.0)
							{
								routeSuccess = false;
								noRouteSegment = false;
								overflowExist = true;
								break;
							}
						}

						if (overflowExist)
						{
							logs << "source : overflow during via insertion" << endl;
							break;
						}
						//sourceViaSuccess = true;
						//routeSuccess = true;//as the via gets pushed into the segment list only if routeSuccess is true
						//Above lines not useful, as the loop continues further without reaching the if(sourceViaSuccess)
						sourceViaSegRoute = new Route(source->row, source->col, source->lay,
													  source->row, source->col, minLayer, ALONG_Z, N);
						N->segmentList.push_back(sourceViaSegRoute);

						/*
						//logs << "via before : " << getCongestion(3, 3, 1) << endl;
						sourceViaSegRoute = new Route(source->row, source->col,
													  source->lay, source->row, source->col, minLayer,
													  ALONG_Z, N);
						//why not overflow check present for the via being inserted here? _PRJ

						N->segmentList.push_back(sourceViaSegRoute);
						//netWirelengthDemand(N, true, false);
						noRouteSegment = false;
						routeSuccess = true;
						//if (allPinsInSameLayer)
							//i++;
						//logs << "via after : " << getCongestion(3, 3, 1) << endl;
						//continue;
*/
					}
				}
				else //if minLayerCstr is false, but pins are in the same grid
				{
					if (segmentPresent == true)
					{
						//netWirelengthDemand(N, true, false);
						noRouteSegment = true;
						routeSuccess = true;
						//if (allPinsInSameLayer)
						//	i++;
						//continue;
					}
					else //if segmentPresent == false
					{
						//netWirelengthDemand(N, true, false);
						noRouteSegment = true;
						routeSuccess = false;
						//if (allPinsInSameLayer)
						//	i++;
						//continue; //moves to next iteration of the loop, i.e. the next pin
					}
				}

				//if (allPinsInSameLayer)
				//i++;
				//logs << "via after : " << getCongestion(3, 3, 1) << endl;
				continue;
			}
			else
			{
				noRouteSegment = false;
			}

			/*			if (pinsInsameGrid && minLayerCstr == true && minLayer == source->lay && segmentPresent == true)
			{
				//netWirelengthDemand(N, true, false);
				noRouteSegment = true;
				routeSuccess = true;
				// if (allPinsInSameLayer)
				// 	i++;
				continue;
			}
			else if (pinsInsameGrid && minLayerCstr == true && minLayer == source->lay && segmentPresent == false)
			{
				//netWirelengthDemand(N, true, false);
				noRouteSegment = true;
				// if (allPinsInSameLayer)
				// 	i++;
				continue; //moves to next iteration of the loop, i.e. the next pin
			}
			else if (pinsInsameGrid && minLayerCstr == true && minLayer != source->lay)
			{
				//logs << "via before : " << getCongestion(3, 3, 1) << endl;
				sourceViaSegRoute = new Route(source->row, source->col,
											  source->lay, source->row, source->col, minLayer,
											  ALONG_Z, N);
				N->segmentList.push_back(sourceViaSegRoute);
				//netWirelengthDemand(N, true, false);
				noRouteSegment = false;
				routeSuccess = true;
				// if (allPinsInSameLayer)
				// 	i++;
				//logs << "via after : " << getCongestion(3, 3, 1) << endl;
				continue;
			}
			else
			{
				noRouteSegment = false;
			}
*/

			if (getCongestion(source->row, source->col, source->lay) >= 1)
			{
				routeSuccess = false;
				if (segmentPresent)
					noRouteSegment = false;
				break;
			}

			if (!pinsInsameGrid && source->lay < minLayer)
			{
				bool overflowExist = false;
				for (unsigned i = source->lay; i <= minLayer; i++)
				{
					if (getCongestion(source->row, source->col, i) * 100.0 >= 100.0)
					{
						routeSuccess = false;
						noRouteSegment = false;
						overflowExist = true;
						break;
					}
				}

				if (overflowExist)
				{
					logs << "source : overflow during via insertion" << endl;
					break;
				}
				sourceViaSuccess = true;
				sourceViaSegRoute = new Route(source->row, source->col,
											  source->lay, source->row, source->col, minLayer,
											  ALONG_Z, N);
				/*netWirelengthDemand(N, true, true);
				 N->segmentList.push_back(sourceViaSegRoute);
				 netWirelengthDemand(N, true, false);*/

				//source->lay = minLayer;
				source = gv.dd.gGrid_Route[minLayer][source->row][source->col];
			}

			if (!pinsInsameGrid && destination->lay < minLayer)
			{
				bool overflowExist = false;
				for (unsigned i = destination->lay; i <= minLayer; i++)
				{
					if (getCongestion(destination->row, destination->col, i) * 100.0 >= 100.0)
					{
						routeSuccess = false;
						noRouteSegment = false;
						overflowExist = true;
						break;
					}
				}

				if (overflowExist)
				{
					logs << "destination : overflow during via insertion"
						 << endl;
					break;
				}
				destViaSuccess = true;
				destViaSegRoute = new Route(destination->row, destination->col,
											destination->lay, destination->row, destination->col,
											minLayer, ALONG_Z, N);
				/*netWirelengthDemand(N, true, true);
				 N->segmentList.push_back(destViaSegRoute);
				 netWirelengthDemand(N, true, false);*/
				//destination->lay = minLayer;
				destination = gv.dd.gGrid_Route[minLayer][destination->row][destination->col];
			}

			if (getCongestion(destination->row, destination->col,
							  destination->lay) >= 1)
			{
				routeSuccess = false;
				if (segmentPresent)
					noRouteSegment = false;
				break;
			}
			wavefront.push(source);
			gv.dd.gGrid_Route[source->lay][source->row][source->col] = source;
			//gv.dd.q.push(source);
			wirelengthCost = 0;
			auto findRouteStart = std::chrono::system_clock::now();
			while (!wavefront.empty())
			{
				auto findRouteFinish = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsedFindRoute = (findRouteFinish - findRouteStart);
				double time_taken = elapsedFindRoute.count();
				//logs << ctime(&moveCellFinish) << endl;

				time_taken /= 60;
				if (time_taken >= 60)
				{
					logs << "In Find Route, Duration >= " << time_taken << " s" << endl;
					routeSuccess = false;
					noRouteSegment = false;
					break;
				}
				routeGuide *cell = wavefront.top();
				wavefront.pop();
				//routeGuide *cell = gv.dd.q.front();
				cell->expanded = true;
				gv.dd.gGrid_Route[cell->lay][cell->row][cell->col]->expanded =
					true;
				/*logs << " row: " << cell->row << " col: " << cell->col << " lay: "
				 << cell->lay << " popped" << endl;*/

				if (isEqual(cell, destination))
				{
					logs << "target reached 1" << endl;
					backtrack(cell, N, logs);
					initializeRoute(rowStart, rowEnd, colStart, colEnd);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
					routeSuccess = true;
					break;
				}

				//vector<routeGuide *> nb;
				bool destFound = false;
				destFound = getNbforGivenCell(rowStart, rowEnd, colStart,
											  colEnd, cell, destination, N, minLayer, tempWf, wirelengthCost, logs);
				if (destFound)
				{
					logs << "target reached 2" << endl;
					routeSuccess = true;

					while (!tempWf.empty())
					{
						//routeGuide *oldWf = tempWf.top();
						//logs << "Deleting from temporary wave front" << endl;
						//delete oldWf;
						tempWf.pop();
					}

					break;
				}

				while (!tempWf.empty())
				{
					routeGuide *r = tempWf.top();
					if (isEqual(r, destination))
					{
						logs << "found" << endl;
					}
					r->expanded = true;
					/*logs << " row: " << r->row << " col: " << r->col << " lay: "
					 << r->lay << "pred: " << r->pred << " added" << endl;*/
					wavefront.push(r);
					tempWf.pop();
				}
			}
			if (routeSuccess)
			{
				if (sourceViaSuccess)
				{
					//netWirelengthDemand(N, true, true);
					N->segmentList.push_back(sourceViaSegRoute);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
				}
				if (destViaSuccess)
				{
					//netWirelengthDemand(N, true, true);
					N->segmentList.push_back(destViaSegRoute);
					//netWirelengthDemand(N, true, false); //Calculating and updating the segment demands of the routed net in the 3D array
				}
			}
			// if (allPinsInSameLayer)
			// 	i++;
		}
	}

	//logs << "route possibility : " << routeSuccess << endl;
	if (!noRouteSegment && !routeSuccess)
	{
		logs << "Inside 'routeSuccess Failure' If Block" << endl;
		logs << " for Net: " << N->name << endl;
		//logs << "no route possible : " << routeSuccess << endl;
		//update the segment list
		//netWirelengthDemand(N, true, true);
		for (list<Route *>::iterator i = N->segmentList.begin();
			 i != N->segmentList.end(); i++)
		{
			delete *i;
		}
		N->segmentList.clear();
		/*for (itr1 = gv.dd.routeList.begin(); itr1 != gv.dd.routeList.end();
		 ++itr1)
		 logs << (*(itr1))->net->name << endl;*/
		//Clear the segments for that Net from the route list also, as produce output function uses routeList only
		itr1 = gv.dd.routeList.begin();
		while ((itr1 != gv.dd.routeList.end()) && (*itr1)->net != N)
		{
			//logs << "In Routelist traversal Begin " << (*itr1)->net->name<< endl;
			itr1++;
		}
		itr2 = itr1;
		while ((itr2 != gv.dd.routeList.end()) && (*itr2)->net == N)
		{
			//logs<<"In Routelist traversal End "<<(*itr2)->net->name<<endl;
			N->segmentList.push_back(*itr2);
			/*logs << (*itr2)->sRow << " " << (*itr2)->sCol << " "
				 << (*itr2)->sLayer << " " << (*itr2)->eRow << " "
				 << (*itr2)->eCol << " " << (*itr2)->eLayer << " " << N->name
				 << endl;*/
			itr2++;
		}
		netWirelengthDemand(N, true, false);
		return false;
	}
	if (!routeSuccess && noRouteSegment)
	{
		netWirelengthDemand(N, true, false); //need to update the demand back, but not required to restore the routeList
											 //as there is not any for noRouteSegment == true
		return false;
	}
	if (routeSuccess)
	{
		netWirelengthDemand(N, true, false);
		return true;
	}

	logs.close();

	return routeSuccess;
}

unsigned commitRoute(Net *N, bool commit, ofstream &logs)
{

	vector<Route *>::iterator itr1, itr2;
	if (commit)
	{
		//update the routing list

		/*for(itr1=gv.dd.routeList.begin(); itr1!=gv.dd.routeList.end(); ++itr1)
		 logs<<(*(itr1))->net->name<<endl;*/
		//Clear the segments for that Net from the route list also, as produce output function uses routeList only
		itr1 = gv.dd.routeList.begin();
		while ((itr1 != gv.dd.routeList.end()) && (*itr1)->net != N)
		{

			//logs<<"In Routelist traversal Begin "<<(*itr1)->net->name<<endl;
			itr1++;
		}
		itr2 = itr1;
		while ((itr2 != gv.dd.routeList.end()) && (*itr2)->net == N)
		{
			//delete *itr2;
			itr2++;
		}
		m1.lock();
		gv.dd.routeList.erase(itr1, itr2);
		m1.unlock();
		logs << "Commiting route" << endl;
		for (list<Route *>::iterator i = N->segmentList.begin();
			 i != N->segmentList.end(); ++i)
		{

			logs << (*i)->sRow << " " << (*i)->sCol << " " << (*i)->sLayer
				 << " " << (*i)->eRow << " " << (*i)->eCol << " "
				 << (*i)->eLayer << " " << N->name << endl;
			m1.lock();
			gv.dd.routeList.push_back(*i);
			m1.unlock();
		}
		return netWirelengthDemand(N, false, false);
		/*for (itr1 = gv.dd.routeList.begin(); itr1 != gv.dd.routeList.end();
		 ++itr1)
		 logs << (*(itr1))->net->name << endl;*/
	}
	else
	{

		netWirelengthDemand(N, true, true);
		//update the segment list
		for (list<Route *>::iterator i = N->segmentList.begin();
			 i != N->segmentList.end(); i++)
		{
			delete *i;
		}
		N->segmentList.clear();
		//getDemand();
		/*for (itr1 = gv.dd.routeList.begin(); itr1 != gv.dd.routeList.end();
		 ++itr1)
		 logs << (*(itr1))->net->name << endl;*/
		//Clear the segments for that Net from the route list also, as produce output function uses routeList only
		itr1 = gv.dd.routeList.begin();
		while ((itr1 != gv.dd.routeList.end()) && (*itr1)->net != N)
		{

			/*logs << "In Routelist traversal Begin " << (*itr1)->net->name
			 << endl;*/
			itr1++;
		}
		itr2 = itr1;
		while ((itr2 != gv.dd.routeList.end()) && (*itr2)->net == N)
		{

			//logs<<"In Routelist traversal End "<<(*itr2)->net->name<<endl;
			N->segmentList.push_back(*itr2);
			logs << (*itr2)->sRow << " " << (*itr2)->sCol << " "
				 << (*itr2)->sLayer << " " << (*itr2)->eRow << " "
				 << (*itr2)->eCol << " " << (*itr2)->eLayer << " " << N->name
				 << endl;
			itr2++;
		}
		return netWirelengthDemand(N, true, false);
	}
}
unsigned getDemandsOfCell(Inst *Cell) // ERROR : Why are passing all the blockages when L1 L2 and L3 Blockages need to be considered separately
{
	unsigned cellDemand = 0;
	for (vector<Blockage>::iterator blkgitr = Cell->master->blockageList.begin(); blkgitr != Cell->master->blockageList.end(); blkgitr++)
	{
		cellDemand += (*blkgitr).demand;
	}

	return cellDemand;
}

void dataFileDemandSupplyHeadersCreation(ofstream &predictionLogs)
{
	unsigned i = 1, j = 1, p = 1;

	predictionLogs << "rows,"
				   << "cols,"
				   << "layers,"
				   << "cells,";

	for (i = 1; i <= NUM_CELLS; i++)
	{
		predictionLogs << "row" << i << ","; //row1 --> row pos of cell 1
	}
	for (j = 1; j <= NUM_CELLS; j++)
	{
		predictionLogs << "col" << j << ",";
	}
	for (i = 1; i <= NUM_CELLS; i++)
	{
		predictionLogs << "no_pins_" << i << ",";
	}
	for (i = 1; i <= NUM_CELLS * MAX_PINS_FOR_ANY_CELL; i++)
	{
		predictionLogs << "MSB1connection_pid" << i << ",";
		predictionLogs << "LSB1connection_pid" << i << ",";
		predictionLogs << "MSB2connection_pid" << i << ",";
		predictionLogs << "LSB2connection_pid" << i << ",";
	}
	for (p = 1; p <= NUM_CELLS; p++)
	{
		predictionLogs << "demand" << p << ",";
	}
	for (p = 1; p <= NUM_LAYERS; p++)
	{
		for (i = 1; i <= NUM_ROWS; i++)
		{
			for (j = 1; j <= NUM_COLS; j++)
			{
				predictionLogs << "supply-demand" << i << "_" << j << "_" << p << ",";
				//dataFile<<"demand"<<i<<"_"<<j<<"_"<<p<<",";
			}
		}
	}
	if (GENERATE_CELL_CONNECTIVITY_IN_DATAFILE)
	{
		for (i = 1; i <= NUM_CELLS; i++)
		{
			/*
        for(j=1;j<=NUM_CELLS;j++)
        {
            dataFile<<"connection"<<i<<"_"<<j<<",";
        }*/
			predictionLogs << "connectionsOf" << i << ",";
		}
	}
	predictionLogs << "movedCellIndex"
				   << ","; //cellIndex to be moved
	predictionLogs << "mCellRow"
				   << ","; //current row location of cell to be moved
	predictionLogs << "mCellCol"
				   << ","; //current col location of cell to be moved
	predictionLogs << "mCellDemand"
				   << ","; //current col location of cell to be moved
	predictionLogs << "movedCellRowPos"
				   << ","; //location to which cell moved to in ideal solution
	predictionLogs << "movedCellColPos"
				   << ",";
	predictionLogs << "cellName"
				   << ",";
	predictionLogs << "movedFlag"
				   << ",";

	predictionLogs << endl;
}
unsigned printRandoms(int lower, int upper)
{
	unsigned num = (rand() % (upper - lower + 1)) + lower;
	return num;
}
bool checkIfCellMovable(Inst *Cell, ofstream &logs)
{
	bool movable = false;
	//    Net* cellNet;

	if (Cell->movable == true)
	{
		movable = true;
		for (vector<Net *>::iterator n = (Cell->netList).begin();
			 n != (Cell->netList).end(); ++n)
		{
			if ((*n)->inFixedPartition == true)
			{
				movable = false;
				Cell->inFixedPartition = true;
				m3.lock();
				gv.dd.instListFixedPartition.push_back(Cell);
				m3.unlock();
				break;
			}
		}
	}
	//	logs << Cell->name << " " << movable << endl;
	return movable;
}
unsigned partitionMoveCell(vector<Inst *> instList, unsigned rowBegIdx, //  This instList needs to be passed to the sliding window scan.
						   unsigned rowEndIdx, unsigned colBegIdx, unsigned colEndIdx,
						   unsigned &numCellsMoved, double maxTime, ofstream &logs) // Call sliding/Window/Scan from inside this
{
	bool cellMovable = false;
	auto moveCellStart = std::chrono::system_clock::now();
	float congestion = 1.0;
	priority_queue<Inst *, vector<Inst *>, CompareCellCongestion> prioritizedInstList;

	for (vector<Inst *>::iterator itr = instList.begin(); itr != instList.end(); ++itr)
	{
		cellMovable = checkIfCellMovable(*itr, logs); //to check if cell belongs to fixed instList
	}

	// list would be ready after this. You just have to do the processing now.
}
float getStepForWindow(float num, float near)
{
	//cout << "num = " << num << endl;
	if ((int)((int)num % (int)2) != 0) //odd
		num++;
	//cout << "num = " << num << endl;
	//exit(0);
	float i = near;
	while (1)
	{
		if ((int)((int)num % (int)i) == 0)
			break;

		i++;
	}
	if (i > 10)
		i = 5;

	return i;
}

int findStepSizeAboveAndBelow(int number, unsigned min, unsigned max)
{
	bool prime = isPrime(number);
	cout << "Number is prime ? =>" << prime << endl;

	int multiple;
	if (!prime)
	{
		multiple = findMultipleGreaterThanOneAbove(number, min);
		cout << "Multiple of " << number << " above " << min << " was found to be " << multiple << endl;
	}
	else
	{
		int n = number - 1;
		multiple = findMultipleGreaterThanOneAbove(n, min);
		cout << "Multiple of " << number << " above " << min << " was found to be " << multiple << endl;
	}

	if (multiple > max)
	{
		cout << "But is greater than the maximum step size allowed" << endl;
		int n = number - 1;
		if (!isPrime(n))
			multiple = findMultipleGreaterThanOneAbove(n, min);
		else
		{
			cout << "Result here too came out to be prime #" << endl;
			n--;
			multiple = findMultipleGreaterThanOneAbove(n, min);
		}
		cout << "Multiple of " << n << " above " << min << " was found to be " << multiple << endl;
	}

	if (multiple)
		cout << "I will lose the dimension by : " << number % multiple << endl;
	else
		multiple = 5;
	return multiple;
}

int findMultipleGreaterThanOneAbove(unsigned number, unsigned min)
{

	for (int i = min; i <= number / 2; i++)
	{
		if (number % i == 0)
			return i;
	}

	return 0;
}

bool isPrime(unsigned number) // copy this simply (This is reliable)
{
	bool primeFlag = true;
	for (int i = 2; i <= number / 2; i++)
	{
		if (number % i == 0)
			primeFlag = false;
	}
	return primeFlag;
}

float mean(float *X, int N)
{
	long long acc = 0;
	for (unsigned i = 0; i < N; i++)
	{
		acc += X[i];
	}
	float m = (float)acc / N;
	return m;
}

float standardDeviation(float *X, int N)
{
	float m = mean(X, N);
	float standardDeviation = 0.0;
	for (unsigned i = 0; i < N; ++i)
	{
		standardDeviation += pow(X[i] - m, 2);
	}
	return sqrt(standardDeviation / N);
}

void standardScaler(float *X, unsigned N)
{
	float m = mean(X, N);
	float sd = standardDeviation(X, N);
	for (unsigned i = 0; i < N; i++)
	{
		X[i] = (X[i] - m) / sd;
	}
}

int partitionDesignClustered(vector<Inst *> &instList, ofstream &dataFile, unsigned rowStartIndex, unsigned colStartIndex, unsigned rowEndIndex, unsigned colEndIndex,
							 unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
							 list<Inst *> &cellsInWindow, list<Inst *> &cellsInRowWindow, list<Inst *> &cellsBeyondRowWindow,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials,const fdeep::model &model)
{ // ? HEre e pass only the present window after sliding by a predetermined amount
	
	
	unsigned cellRow = 0, cellCol = 0;
	//unsigned numPinsArray[NUM_CELLS + 1];
	struct pinIndexes pinIndexList[10 + 1];
	unsigned cellIdxList[10 + 1];		// Done
	unsigned rowPosCell[10 + 1];			// Done
	unsigned colPosCell[10 + 1];			// Done
	unsigned demandValCell[10 + 1];		// Done
	unsigned movableCellIdxList[10 + 1]; // Done
	unsigned numPinsArray[10+ 1];		// Done
	//bool connectivity_matrix[2001][2001];		// Done
	unsigned i=1;
	unsigned k = 0; // numCells
	list<Inst *> movableCellList;
	clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); //commented as being called inside generateDataSample

	list<Inst *>::iterator itr = /*gv.*/ cellsInRowWindow.begin();
	//bool pin_connectivity_matrix[10 * 32][10 * 32];

	//vector <Inst *>::iterator itr;
	unsigned cellIdx = 0;
	//cout << "Window: " << startGridRowIdx << " " << endGridRowIdx << " " << startGridColIdx << " " << endGridColIdx << endl;
	//clearPinConnectivityMatrix(pin_connectivity_matrix);
	// numPredictionTrials += 1;
	// if(difftime(time(NULL), start) > 120){
	// 	return -1;
	// }
	// return 1;
	if (startGridRowIdx == /*gv.rowBeginIdx*/ rowStartIndex && startGridColIdx == /*gv.colBeginIdx*/ colStartIndex) // *  first time partition, starting at origin
	{
		for (vector<Inst *>::iterator instListItr = /*gv.dd.*/ instList.begin(); instListItr != /*gv.dd.*/ instList.end(); ++instListItr)
		{
			//cout<<"Inside for loop 1"<<endl;
			// cellRow = (*instListItr)->row;
			// cellCol = (*instListItr)->col;

			// cout<<"Cell: "<<(*instListItr)->name<<endl;
			if ((*instListItr)->row >= startGridRowIdx && (*instListItr)->row <= endGridRowIdx) // ?  The row window check
			{
				/*gv.*/ cellsInRowWindow.push_back(*instListItr);
				//cout<<" is in row window "<<startGridRowIdx<<" "<<endGridRowIdx<<endl;

				if ((*instListItr)->col >= startGridColIdx && (*instListItr)->col <= endGridColIdx) // ? The column window check
				{
					cellIdx++;
					// * Cell index is incremental only
					// ? But since we make the instance list while parsing then the cells in the list come in order of their names.
					// ? This means that though the names would be in increasing order but it is not necessary for this to be true
					// ? in the sliding window. But, one thing is true that the cell index if not 0 then would be incremental only.
					// TODO :  Check that you are using cellIdx for calculating the pin index.
					(*instListItr)->cellIndex = cellIdx;
					/*gv.*/ cellsInWindow.push_back(*instListItr);
					// cout<<" and in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
				}
				else
				{
					(*instListItr)->cellIndex = 0; // Inside the row window but outside Window
				}
			}

			else
			{
				/*gv.*/ cellsBeyondRowWindow.push_back(*instListItr); // Outside the row window and window both
				(*instListItr)->cellIndex = 0;
				//cout<<" is NOT in row window "<<startGridRowIdx<<" "<<endGridRowIdx<<endl;
			}
			// ? We would have cellIdx = 0 for any cell that is beyond the current cliding window(cell not exixting inside cellsInRowWindow and cellsInWindow).
			/*
            if( cellCol >= startGridColIdx && cellCol <= endGridColIdx)
            {
                gv.cellsInColWindow.push_back(*instListItr);
                cout<<" is in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
            }
            else
            {
                gv.cellsBeyondColWindow.push_back(*instListItr);
                cout<<" is NOT in col window "<<startGridColIdx<<" "<<endGridColIdx<<endl;
            }*/
		}
	}
	else // * If not for first time.
	{
		//logs << "clearing cells in window list" << endl;
		/*gv.*/ cellsInWindow.clear(); // ? After each slide, the cellsInWindow needs to be emptied
		//		if(gv.cellsInRowWindow.empty())
		//		cout<<"Cells in Row Window list is empty"<<endl;
		for (itr = /*gv.*/ cellsInRowWindow.begin(); itr != /*gv.*/ cellsInRowWindow.end(); ++itr)
		{
			//	cout<<"Inside for loop"<<endl<<(*itr)->name;
			// cellRow = (*itr)->row;
			// cellCol = (*itr)->col;
			//cout<<"Cell row col accessed"<<endl;

			if ((*itr)->row < startGridRowIdx || (*itr)->row > endGridRowIdx)
			{
				(*itr)->cellIndex = 0; // ? Delete the cell from the row window if not in the new row bounds
									   //cout<<"Removing from cells in row window"<<endl;
									   //cout<<"Cell: "<<(*itr)->name<<endl;

				//gv.cellsInRowWindow.remove(*itr);
			}
			else
			{
				if ((*itr)->col >= startGridColIdx && (*itr)->col <= endGridColIdx)
				{
					cellIdx++; // * Again, the cell indexes are incremental
					(*itr)->cellIndex = cellIdx;
					//	cout<<"Adding cell in window"<<endl;
					/*gv.*/ cellsInWindow.push_back(*itr); // Put in the row window
				}
				else
				{
					(*itr)->cellIndex = 0; // ! DOUBT
				}
			}
		}
		if (startGridColIdx == /*gv.colBeginIdx*/ colStartIndex) //=> row window cell list to be updated as per new row window limits
		{														 // ! DOUBT
			//cout<<"=> row window cell list to be updated as per new row window limits"<<endl;

			for (itr = /*gv.*/ cellsBeyondRowWindow.begin(); itr != /*gv.*/ cellsBeyondRowWindow.end(); ++itr)
			{
				// cellRow = (*itr)->row;
				// cellCol = (*itr)->col;

				if ((*itr)->row >= startGridRowIdx && (*itr)->row <= endGridRowIdx) // ? Checking if the cell is in current row window
				{
					/*gv.*/ cellsInRowWindow.push_back(*itr);

					if ((*itr)->col >= startGridColIdx && (*itr)->col <= endGridColIdx)
					{
						cellIdx++;
						(*itr)->cellIndex = cellIdx;
						/*gv.*/ cellsInWindow.push_back(*itr);
					}
					else
					{
						(*itr)->cellIndex = 0;
					}
					//gv.cellsBeyondRowWindow.remove(*itr);
				}
			}
		}
	}
	// TODO : Check if the cell window being captured are correct or not
	// numPredictionTrials += 1;
	// if(difftime(time(NULL), start) > 120){
	// 	return -1;
	// }
	// return 1;
	
	logs << "Total number of cells in window: " << cellIdx << endl;
	if (cellIdx > 2000)
	{
		logs << "Returning from function as number of cells > 2000" << endl;
		return 0;
	}
	else if (cellIdx == 0)
	{
		logs << "Returning from function as there are no cells in the window" << endl;
		return 0;
	}

	//populate connectivity matrix for this window and generate data samples
	// ? ******************************************************* CELLS IN WINDOW CREATED, NOW PASSING TO MODEL STARTS ******************************************
	list<Inst *>::iterator adjCellItr; //, findItr;
	unsigned cellCount = 0, currentCellCount = 0, movableCellCount = 0;
	unsigned numNetsCurrentCell;
	bool currentCellAccepted = false;
	bool cellPresentInCellIdxList = false;
	Inst *currentCell;
	unsigned numPins = 0;
	unsigned net_connectivity_matrix[10 + 1][10 + 1];
	for (list<Inst *>::iterator cellIterator = cellsInWindow.begin(); cellIterator != cellsInWindow.end(); cellIterator++)
	{
		currentCell = (*cellIterator);
		if ((currentCell)->adjacencyList.size() < 10)
		{
			currentCellAccepted = true;
			cellCount++;
			cellIdxList[cellCount] = currentCell->cellIndex;
			movableCellIdxList[cellCount] = currentCell->cellIndex;
			rowPosCell[cellCount] = currentCell->row - startGridRowIdx + 1;
			colPosCell[cellCount] = currentCell->col - startGridColIdx + 1;
			demandValCell[cellCount] = returnL1Blockage(currentCell);
			movableCellList.push_back(currentCell);
			for (list<Inst *>::iterator adjacentCell = currentCell->adjacencyList.begin(); adjacentCell != currentCell->adjacencyList.end(); adjacentCell++)
			{
				Inst *adjCell = (*adjacentCell);
				if (adjCell->cellIndex != 0)
				{
					cellCount++;
					cellIdxList[cellCount] = adjCell->cellIndex;
					movableCellIdxList[cellCount] = currentCell->cellIndex;
					rowPosCell[cellCount] = adjCell->row - startGridRowIdx + 1;
					colPosCell[cellCount] = adjCell->col - startGridColIdx + 1;
					demandValCell[cellCount] = returnL1Blockage(adjCell);

					movableCellList.push_back(adjCell);
				}
				else
				{
					currentCellAccepted = false;

					break;
				}
			}
			if (currentCellAccepted)
			{
					//cout <<"Col Window: "<<startGridColIdx<<" "<<endGridColIdx<<endl;
				
					unsigned numNets = createNetBasedConnectivityMatrix(cellCount, cellIdxList, net_connectivity_matrix, movableCellList);
					int flag = generateOutputDataSampleCurrentCell(dataFile, rowStartIndex, rowEndIndex, colStartIndex,
														colEndIndex, startGridRowIdx, endGridRowIdx, startGridColIdx,
														endGridColIdx, net_connectivity_matrix, rowPosCell, colPosCell,
														cellIdxList, demandValCell, movableCellIdxList, numPinsArray,
														pinIndexList, movableCellList, cellCount, numNets,logs,numCellsMoved,numPredictionTrials,model);
					if(flag==1) return -1;

					cellCount = 0;
					clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); // Clear all the arrays in this functions
					movableCellList.clear();	
			}
			else
			{
				cellCount = 0;
				clearCellArrays(cellIdxList, rowPosCell, colPosCell, demandValCell, movableCellIdxList, numPinsArray, pinIndexList); // Clear all the arrays in this functions
				movableCellList.clear();
			}
		}
	}
	return 1;
}
void clearNetConnectivityMatrix(unsigned net_connectivity_matrix[][MAX_NETS + 1])
{
	for (unsigned i = 0; i <= NUM_CELLS; i++)
	{
		for (unsigned j = 0; j <= MAX_NETS; j++)
		{
			net_connectivity_matrix[i][j] = 0;
		}
	}
}
unsigned createNetBasedConnectivityMatrix(unsigned numCells, unsigned *cellIdxList, unsigned net_connectivity_matrix[][10 + 1], list<Inst *> &movableCellList)
{
	// Make sure that this returns the numnets too
	/*
	Decision of net index: -
	1. Only the nets that are connected to the cell 1 should be considered.
	2. Since the condition for current design to be valid is only of there is not adjacent cell that is
	   out of the current window hence all the nets that would be on the pins of Cell 1 would also have the
	   connections inside the window itself.
	*/
	std::unordered_set<string> pinNetList;
	pinNetList.clear();
	clearNetConnectivityMatrix(net_connectivity_matrix);
	list<Inst *>::iterator currentCell = movableCellList.begin();
	Inst *cellOne = (*currentCell);

	unsigned numNets = setResetInternalNetindex(cellOne, pinNetList, true, false);
	if (numNets > 10)
	{
		cout << "Error :: Numnets here came above 10 " << endl;
		exit(0);
	}
	//return numNets;
	for (unsigned i = 1; i <= numCells; i++)
	{
		Inst *movableCell = returnInstFromMovableCellList(cellIdxList[i], movableCellList);
		for (unordered_map<string, Pin *>::iterator pItr = movableCell->pinMap.begin(); pItr != movableCell->pinMap.end(); pItr++)
		{
			Pin *currentPin = (*pItr).second;
			if (currentPin->net->internalNetIndex != 0)
			{
				unsigned netIndex = currentPin->net->internalNetIndex;
				net_connectivity_matrix[i][netIndex] |= 1;
			}
		}
	}
	unsigned trashVar = setResetInternalNetindex(cellOne, pinNetList, false, true);
	// netConnectivity matrix has been generated
	return numNets;
}
Inst *returnInstFromMovableCellList(unsigned cellIdx, list<Inst *> &movableCellList)
{
	for (list<Inst *>::iterator mCellIterator = movableCellList.begin(); mCellIterator != movableCellList.end(); mCellIterator++)
	{
		if ((*mCellIterator)->cellIndex == cellIdx)
		{
			return (*mCellIterator);
		}
	}
}
unsigned setResetInternalNetindex(Inst *cellOne, std::unordered_set<string> &pinNetList, bool set, bool reset)
{
	pinNetList.clear();
	unsigned numNetsCellOne = 0; // thiswould also be the number of cells that would be passed
	for (unordered_map<string, Pin *>::iterator pItr = cellOne->pinMap.begin(); pItr != cellOne->pinMap.end(); pItr++)
	{
		Pin *currentPin = (*pItr).second;
		if (pinNetList.find(currentPin->net->name) == pinNetList.end())
		{
			if (set)
				numNetsCellOne++;
			currentPin->net->internalNetIndex = numNetsCellOne; // set the internal Id
			if (set)
				pinNetList.insert(currentPin->net->name);
		}
	}

	return numNetsCellOne;
}
unsigned returnL1Blockage(Inst *cell)
{
	unsigned blockageL1 = 0;
	for (vector<Blockage>::iterator bitr = cell->master->blockageList.begin(); bitr != cell->master->blockageList.end(); bitr++)
	{
		if ((*bitr).layer == 1)
		{
			blockageL1 = (*bitr).demand;
			break;
		}
	}
	return blockageL1;
}
int generateOutputDataSampleCurrentCell(ofstream &dataFile, unsigned rowStartIndex, unsigned rowEndIndex, unsigned colStartIndex, unsigned colEndIndex,
										 unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx, unsigned endGridColIdx,
										 unsigned net_connectivity_matrix[][MAX_NETS + 1], unsigned *rowPosCell,
										 unsigned *colPosCell, unsigned *cellIdxList, unsigned *demandValCell, unsigned *movableCellIdxList, unsigned *numPinsArray,
										 pinIndexes *pinIndexList, list<Inst *> &movableCellList, unsigned k, unsigned numNetsCurrCell,ofstream &logs,unsigned &numCellsMoved,unsigned &numPredictionTrials,const fdeep::model &model)
{
	double totalTime = 0;
	ofstream pMoveLogs;

	float X[248];
	float Y[2];

	//pMoveLogs.open("predictionMovements.txt", ios::app); // ios::app allows you to write anywhere in file(not just in the end)
	pMoveLogs.close(); //taking lot of disk space
	unsigned i = 1;
	unsigned rowGridStart, colGridStart, rowGridEnd , colGridEnd , mCellGridRow, mCellGridCol ;
	//auto timebefore = chrono::system_clock::to_time_t(chrono::system_clock::now());
	//pMoveLogs << "generateOutputDataSample func started at : " << ctime(&timebefore) << endl;
	list<Inst *>::iterator itr = movableCellList.begin();
	//Initializing the X vector
	Inst *movableCell = (*(movableCellList.begin()));
	std::unordered_set<string> pinNetList;
	clearDataSample(X);
	pinNetList.clear();

	if (movableCell->cellIndex != 0) //: Commented for unit testing, passing cell3 only (one sample case)
	//if (movableCell->cellIndex == FIXED_CELL_IDX)
	{

		unsigned blockageL1 = returnL1Blockage(movableCell);
		if (maxNetsPerCell < numNetsCurrCell)
		{
			maxNetsPerCell = numNetsCurrCell;
		}
		logs << "Cell : " << movableCell->name << endl;
		logs << endl
			 << "Total Blockage for the cell in L1 layer : " << blockageL1;
		logs << endl
			 << "Blockage demand + net demand = " << numNetsCurrCell + blockageL1;

		
		//cout <<"Col Window: "<<startGridColIdx<<" "<<endGridColIdx<<endl;
		generateDataSampleCurrentCell(dataFile, movableCellIdxList[i], net_connectivity_matrix, numNetsCurrCell,
	 							   startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, rowPosCell, colPosCell,
		 							   cellIdxList, demandValCell, movableCellIdxList, numPinsArray, pinIndexList, k, X, Y, false);
		//float X_trial[] = {10, 10, 10, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1, 5, 7, 5, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16777220, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 1, 3, 1, 5, 5, 3, 0, 7, 5, 4, 10, 0, 1, 1, 0, 0, 4, 3, 3, 10, 0, 2, 8, 0, 7, 6, 4, 0, 0, 6, 10, 10, 2, 4, 2, 5, 0, 10, 10, 1, 0, 0, 0, 4, 1, 5, 8, 1, 2, 5, 2, 10, 7, 0, 1, 2, 5, 2, 10, 7, 10, 0, 7, 4, 0, 6, 3, 2, 0, 8, 3, 0, 2, 0, 5, 10, 1, 0, 8, 10, 0, 0, 0, 2, 0, 10, 0, 2, 6, 0, 0, 4, 0, 6, 0, 6, 10, 3, 8, 9, 0, 3, 0, 2, 4, 1, 3, 4, 1, 1, 1};

		pythonPredictionCode(X,Y,model);
		numPredictionTrials += 1;
		// if(difftime(time(NULL), start) > 120){
		// 	return 1;
		// }
		// i++;
		// return -1;
		
		trial++;

		if ((round(Y[0]) > 10) || (round(Y[0]) < 1))
		{
			Y[0] = printRandoms(1, 10);
			numRandomPlacementsGeneratedY1++;
		}

		if ((round(Y[1]) > 10) || (round(Y[1]) < 1))
		{
			Y[1] = printRandoms(1, 10);
			numRandomPlacementsGeneratedY2++;
		}

		rowGridStart = rowStartIndex; // gv.rowBeginIdx;
		rowGridEnd = rowEndIndex;	  // gv.rowEndIdx;
		colGridStart = colStartIndex; //  gv.colBeginIdx;
		colGridEnd = colEndIndex;	  // gv.colEndIdx;

		mCellGridRow = round(Y[0]) + startGridRowIdx - 1;
		mCellGridCol = round(Y[1]) + startGridColIdx - 1;

		unsigned nMCells = 0;

		unsigned tempMCellGridRow = mCellGridRow;
		unsigned tempMCellGridCol = mCellGridCol;
		unsigned restoreRow = mCellGridRow;
		unsigned restoreCol = mCellGridCol;
		unsigned numTrials = 0;
		unsigned tempNMCells = numCellsMoved;

		nMCells = pMoveCell(movableCell, rowGridStart, rowGridEnd, colGridStart, colGridEnd, tempMCellGridRow, tempMCellGridCol, pMoveLogs,numCellsMoved);
		// //logs << "Number of cells moved in this partition are " << nMCells << endl;
		int overflow = 0;
		for (unsigned lay = 1; lay <= gv.numLayers && overflow == 0; lay++)
			for (unsigned row = 1; row <= gv.rowEndIdx && overflow == 0; row++)
				for (unsigned col = 1; col <= gv.colEndIdx && overflow == 0; col++)
				{
					overflow += getOverFlow(row, col, lay);
					if (overflow > 0)
					{

						break; //???????????????????????
					}
				}
		if (overflow)
		{
			logs << "Överflow occured here" << endl;
			// exit(0);
		}
		// Take care of the global variales that are updated due to thhis
		tempMCellGridRow = restoreRow;
		tempMCellGridCol = restoreCol;

		logs << "If moved then moved to " << tempMCellGridRow << " " << tempMCellGridCol << endl;
		if (i > 10)
			logs << "Error 2: MovableCellIdxList index >10" << endl;
		i++;
		if(difftime(time(NULL), start) >120 ){
			return 1;
		}
		

// 		 if (elapsed.count() > 120) // 6 hours
// 		 {

// 		 	logs.open("logx.txt"+outFile);
// 		 	logs << "Did not complete the scan 1 itself even after fixed time" << endl;
// 		 	logs << "Number of times repeated : " << numRepeat << endl;
// 		 	logs << "\nDuration of running :" << elapsed.count() << "m" << endl;
// 		 	logs << "Number of times repeated : " << numRepeat << endl;
// 		 	logs << "Maximum nets per cell : " << maxNetsPerCell << endl;
// 		 	logs << "numMovedCells = " << numCellsMoved << endl;
// 		 	logs << "numPredictionTrials = " << numPredicitionCalls << endl;


// 		 	logs << "---------------- Acceptance Rate information ---------------------" << endl;
// 		 	logs << "Maximum Nets per cell = " << maxNetsPerCell << endl;
// 		 	logs << "numMovedCells = " << numCellsMoved << endl;
// 		 	logs << "numPredictionTrials = " << numPredicitionCalls << endl;
// 		 	logs.close();
// 		 	//produceRouterLogs();
// 		 	//gv.dd.produceOutput(outFile);
// 		 	//exit(0);
// 			// flag = 1;
// 		 	//break;
// 			return 1;
// 		 }*/
// 		//initialize input vector X for further iterations
// 		//clearCellArrays();
// 		//generateDataSample(dataFile, 0, startGridRowIdx, endGridRowIdx, startGridColIdx, endGridColIdx, true);
     }
 	 auto timeafter = chrono::system_clock::to_time_t(chrono::system_clock::now());
 	 pMoveLogs << "generateOutputDataSample funct finished at : " << ctime(&timeafter) << endl;
 	// // return flag
	 return -1;

}

void generateDataSampleCurrentCell(ofstream &dataFile, unsigned movableCellIndex, unsigned net_connectivity_matrix[][MAX_NETS + 1],
								   unsigned numNetsCurrCell, unsigned startGridRowIdx, unsigned endGridRowIdx, unsigned startGridColIdx,
								   unsigned endGridColIdx, unsigned *rowPosCell, unsigned *colPosCell, unsigned *cellIdxList, unsigned *demandValCell,
								   unsigned *movableCellIdxList, unsigned *numPinsArray, pinIndexes *pinIndexList, unsigned k, float *X, float *Y,
								   bool initialize)
{ // iniitalize variable is passed as first

	//PyObject *pValue2;
	
	unsigned connectivityRegister = 0x01;
	unsigned cellIndex = 0, connectedCellIndex = 0;
	unsigned bitVector;
	unsigned i = 1, j = 1;

	unsigned movedCellLocation = 0;

	unsigned inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL;

	std::memset(X, 0, sizeof(X));

	X[PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL] = gv.n;
	X[PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL] = gv.m;
	X[PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL] = k;
	// X[PYTHON_INPUT_VECTOR_PROBLEM_INDEX] = FIXED_PROBLEM_INDEX;
	X[PYHTON_INPUT_VECTOR_NUMNETS] = numNetsCurrCell;

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL;
	for (i = 1; i <= k; i++)
	{
		if ((initialize == false) && (cellIdxList[i] == movableCellIndex))
		{
			movedCellLocation = i;
		}
		X[inputVectorCountML] = ((initialize == false) ? rowPosCell[i] : 0);
		if ((rowPosCell[i] == 0) && (initialize == false))
		{
			// cout << "ERROR : Row position cannot be zero.\nFound it to be zero at i = " << i << endl
			// 	 << "Numcells = " << k << endl;
			;
		}

		inputVectorCountML++;
	}

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_COLPOS_INDEX_VAL;
	for (j = 1; j <= k; j++)
	{
		X[inputVectorCountML] = ((initialize == false) ? colPosCell[j] : 0);
		if ((colPosCell[j] == 0) && (initialize == false))
		{
			// cout << "ERROR : Col position cannot be zero.\nFound it to be zero at j = " << j << endl
			// 	 << "Numcells = " << k << endl;
			;
		}

		inputVectorCountML++;
	}
	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_NET_CONNECTIVITY_INDEX_VAL;
	unsigned long decimalNumber = 0;
	double p = 0;
	for (unsigned i = 1; i <= NUM_CELLS; i++)
	{
		for (unsigned j = 1; j <= MAX_NETS; j++)
		{
			decimalNumber += net_connectivity_matrix[i][j] * pow((double)2, (double)(MAX_NETS - p - 1));
			p++;
		}
		p = 0;

		X[inputVectorCountML] = ((initialize == false) ? (double)decimalNumber / NET_CONNECTIVITY_SCALING_FACTOR : 0);
		inputVectorCountML++;
		decimalNumber = 0;
	}

	inputVectorCountML = PYTHON_INPUT_VECTOR_CELL_DEMAND_INDEX_VAL;
	for (j = 1; j <= k; j++)
	{
		//cout << "Demand put : " << demandValCell[j] << endl;
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ demandValCell[j] : 0);
		inputVectorCountML++;
	}
	//cout <<"Col Window: "<<startGridColIdx<<" "<<endGridColIdx<<endl;
	inputVectorCountML = PYTHON_INPUT_VECTOR_SUPPLY_DEMAND_VAL_INDEX_VAL;
	for (unsigned lay = 1; lay <= gv.l; lay++)
	{
		for (i = startGridRowIdx; i <= endGridRowIdx; i++)
		{
			for (j = startGridColIdx; j <= endGridColIdx; j++)
			{
				
				//cout <<"start col idx: "<<startGridColIdx<<" end col idx: "<<endGridColIdx<<endl;
				if (initialize == false)
				{
					// cout << "layer number = " << lay << endl;
					// cout << "row pos = " << i << " col pos = " << j << endl;

					if (lay > NUM_LAYERS_IN_MODEL){
						goto label1;
					}
					// bool overFlow = false;
					// if (gv.dd.gGrid_supply[lay][i][j] < gv.dd.gGrid_demand[lay][i][j])
					// {
					// 	// cout << "\nERROR : Demand > Supply while data creation " << endl;
					// 	// cout << "This means that there was some wrong movement accepted" << endl;
					// 	// cout << "Grid Point : (l r c)" << lay << " " << i << " " << j << endl;
					// 	// cout << "Supply : " << gv.dd.gGrid_supply[lay][i][j] << endl;
					// 	// cout << "Demand : " << gv.dd.gGrid_demand[lay][i][j] << endl;
					// 	overFlow = true;
					// 	// exit(0);
					// }
					// if (gv.dd.gGrid_supply[lay][i][j] > gv.dd.gGrid_demand[lay][i][j])
					// //  if (!overFlow)
					//   {
					// // // 	// cout << "Lay1 : " << (gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j]) << endl;
					// // // 	// cout << "Lay2 : " << (gv.dd.gGrid_supply[lay + 1][i][j] - gv.dd.gGrid_demand[lay + 1][i][j]) << endl;
					// // // 	// float number = (float)(((gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j]) * 100 ) + (gv.dd.gGrid_supply[lay + 1][i][j] - gv.dd.gGrid_demand[lay + 1][i][j])) / 1000;
					// // // 	// printf("Combiined : %f\n", number);
					//  	// X[inputVectorCountML] = ((initialize == false) ? ((float)(gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j]))/SUP_DEM_SCALING_FACTOR : 0); //  change has been done here
					// 	X[inputVectorCountML] = (float)(((gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j]) * 100 ) + (gv.dd.gGrid_supply[lay + 1][i][j] - gv.dd.gGrid_demand[lay + 1][i][j])) / 1000;
					// // // 	// X[inputVectorCountML] = ((initialize == false) ? (number) : 0); 
					//  }
					// else
					// {
					// 	numOverflowsInDataSampleCreation++;
					// 	X[inputVectorCountML] = 0; // in case there is overflow => change the supp-dem to 0 in data sample
					// }
					if(gv.dd.gGrid_supply[lay][i][j] <= gv.dd.gGrid_demand[lay][i][j]){
						numOverflowsInDataSampleCreation++;
						X[inputVectorCountML] = 0;
					}else{
						//cout << inputVectorCountML << endl;
						X[inputVectorCountML] = ((initialize == false) ? ((float)(gv.dd.gGrid_supply[lay][i][j] - gv.dd.gGrid_demand[lay][i][j]))/SUP_DEM_SCALING_FACTOR : 0); 
					}
				}
				inputVectorCountML++;
			}
		}
	}
	label1:

		inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_INDEX_INDEX_VAL;
		X[inputVectorCountML] = ((initialize == false) ? movedCellLocation : 0);

		inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_ROWPOS_INDEX_VAL;
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ rowPosCell[movedCellLocation] : 0);

		inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_COLPOS_INDEX_VAL;
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ colPosCell[movedCellLocation] : 0);

		inputVectorCountML = PYTHON_INPUT_VECTOR_MCELL_DEMAND_INDEX_VAL;
		X[inputVectorCountML] = ((initialize == false) ? /*gv.*/ demandValCell[movedCellLocation] : 0);

		inputVectorCountML++;
		if (inputVectorCountML != 248)
		{
			// cout << "Error: inputVectorCount = " << inputVectorCountML;
			;
		}
		else
		{
			// cout << endl << "Successfully created the data sample";
			;
		}
			
	return;
}

void produceRouterLogs()
{
	ofstream routerLogs;
	routerLogs.open(outFile + ".routerLogs.txt");
	routerLogs << "outFile :" << outFile << endl;
	routerLogs << "numPredictionCalls :" << numPredicitionCalls;
	routerLogs << endl
			   << "Number of times router failed to establsh connection : " << numRouterFailiures << endl;
	//routerLogs << "Number of times prediction was out of bound and higher wirelength: " << numPredictionFailiures << endl;
	routerLogs << "Failiures due to less wirelength :" << numPredictionFailiuresWirelength << endl;
	routerLogs << "Router failiures due to overflows : " << numRouteOverflowFailiures << endl;
	routerLogs << "numMovedCells : " << gv.numcellsmoved << endl;
	routerLogs << "notfound bettter position (same) : " << numNotFoundBetterSolution << endl;
	routerLogs << "HPWL zero : " << initialHPWLZero << endl;
}

unsigned whichPartition(unsigned row){
	if(row>=firstPartitionStart && row<=firstPartitionEnd) return 1;
	else if(row>=secondPartitionStart && row<=secondPartitionEnd) return 2;
	else if(row>thirdPartitionStart && row<=thirdPartitionEnd) return 3;
	else if(row>=fourthPartitionStart && row<=fourthPartitionEnd) return 4;
	else if(row>=fifthPartitionStart && row<=fifthPartitionEnd) return 5;
	else if(row>=sixthPartitionStart && row<=sixthPartitionEnd) return 6;
	else if(row>=seventhPartitionStart && row<=seventhPartitionEnd) return 7;
	else if(row>=eighthPartitionStart && row<=eighthPartitionEnd) return 8;

	return -1;
}
