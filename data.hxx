#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <queue>

//#include "flute.h"

#define T_FOR_GRID_1TO10 5
#define T_FOR_GRID_10TO50 100
#define T_FOR_GRID_50TO100 105
#define T_FOR_GRID_100TO500 200
#define T_FOR_GRID_500TO2000 300
#define T_DEFAULT 2

#define INITIAL_TEMP 100000
#define TEMPERATURE_ALPHA 0.8
#define RADIUS_ALPHA 0.8

//ML DataGen Defines

#define MIN_LAYER 1
#define NUM_LAYERS 10
#define NUM_ROWS 10
#define NUM_COLS 10
#define NUM_CELLS 10
#define CONGESTION_THRESHOLD 0.40
#define FIXED_PROBLEM_INDEX 1
#define PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL 0
#define PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL 1
#define PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL 2
#define PYTHON_INPUT_VECTOR_PROBLEM_INDEX 3
#define PYHTON_INPUT_VECTOR_NUMNETS 3
#define PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL 4
#define PYTHON_INPUT_VECTOR_CELL_COLPOS_INDEX_VAL 14
#define PYTHON_INPUT_VECTOR_CELL_NET_CONNECTIVITY_INDEX_VAL 24
#define PYTHON_INPUT_VECTOR_CELL_DEMAND_INDEX_VAL 34
#define PYTHON_INPUT_VECTOR_SUPPLY_DEMAND_VAL_INDEX_VAL 44
#define PYTHON_INPUT_VECTOR_MCELL_INDEX_INDEX_VAL 244
#define PYTHON_INPUT_VECTOR_MCELL_ROWPOS_INDEX_VAL 245
#define PYTHON_INPUT_VECTOR_MCELL_COLPOS_INDEX_VAL 246
#define PYTHON_INPUT_VECTOR_MCELL_DEMAND_INDEX_VAL 247

//###########OLD DESIGN MACROS JUST TO CHECK ERRORS#######
/*
#define PYTHON_INPUT_VECTOR_NUM_ROWS_INDEX_VAL 0
#define PYTHON_INPUT_VECTOR_NUM_COLS_INDEX_VAL 1
#define PYTHON_INPUT_VECTOR_NUM_CELLS_INDEX_VAL 2
#define PYTHON_INPUT_VECTOR_CELL_ROWPOS_INDEX_VAL 3
#define PYTHON_INPUT_VECTOR_CELL_COLPOS_INDEX_VAL 13

#define PYTHON_INPUT_VECTOR_CELL_DEMAND_INDEX_VAL 353
#define PYTHON_INPUT_VECTOR_SUPPLY_DEMAND_VAL_INDEX_VAL 363
#define PYTHON_INPUT_VECTOR_MCELL_INDEX_INDEX_VAL 463
#define PYTHON_INPUT_VECTOR_MCELL_ROWPOS_INDEX_VAL 464
#define PYTHON_INPUT_VECTOR_MCELL_COLPOS_INDEX_VAL 465
#define PYTHON_INPUT_VECTOR_MCELL_DEMAND_INDEX_VAL 466

*/
#define PYTHON_INPUT_VECTOR_PIN_CONNECTIONS 33
#define PYTHON_INPUT_VECTOR_CELL_NUMPINS_INDEX_VAL 23

#define NET_CONNECTIVITY_SCALING_FACTOR 100
#define GENERATE_CELL_CONNECTIVITY_IN_DATAFILE 1 // Tells that predictionLogs should have cell connectivity in it
#define GENERATE_SINGLE_CELL_CONNECTIVITY_IN_DATAFILE 1
#define NUM_LAYERS_IN_MODEL 2
#define MAX_PINS_FOR_ANY_CELL 32
#define NUM_SPLITS_PER_ROW 10

#define MAX_NETS 10

using namespace std;

typedef enum direction
{
	ALONG_ROW,
	ALONG_COL,
	ALONG_Z,
	UNDEF
} Direction;

//typedef enum Flag {NONE, DEBUG};

struct Layer
{
	string name;
	unsigned number;
	bool horizontal; // true for horizontal
	unsigned default_supply;

	Layer(string nm, unsigned n, bool h, unsigned ds) : name(nm), number(n), horizontal(h), default_supply(ds)
	{
	}
	Layer(const Layer &l)
	{
		name = l.name;
		number = l.number;
		horizontal = l.horizontal;
		default_supply = l.default_supply;
	}
	void write(ostream &os);
};

struct NonDefaultSupply
{
	unsigned row;
	unsigned column;
	unsigned layer;
	int value;
	NonDefaultSupply(unsigned r, unsigned c, unsigned l, int v) : row(r), column(c), layer(l), value(v)
	{
	}
	NonDefaultSupply(const NonDefaultSupply &nds)
	{
		row = nds.row;
		column = nds.column;
		layer = nds.layer;
		value = nds.value;
	}
	void write(ostream &os);
};

struct Supply
{
	unsigned row;
	unsigned column;
	unsigned layer;
	int value;
	Supply(unsigned r, unsigned c, unsigned l, int v) : row(r), column(c), layer(l), value(v)
	{
	}
	Supply(const Supply &s)
	{
		row = s.row;
		column = s.column;
		layer = s.layer;
		value = s.value;
	}
};

struct sameGGridDemand
{
	unsigned row;
	unsigned column;
	unsigned layer;
	vector<string> pairs;
	unordered_map<string, unsigned> demandPair;

	sameGGridDemand(unsigned r, unsigned c, unsigned l, vector<string> v,
					unordered_map<string, unsigned> u) : row(r), column(c), layer(l), pairs(v), demandPair(u)
	{
	}
	sameGGridDemand(const sameGGridDemand &sgd)
	{
		row = sgd.row;
		column = sgd.column;
		layer = sgd.layer;
		pairs = sgd.pairs;
		demandPair = sgd.demandPair;
	}
};

struct MasterPin
{
	unsigned layer;
	string name;
	unsigned masterPinIndex;
	MasterPin(unsigned mpi, string n, unsigned l) : masterPinIndex(mpi), layer(l), name(n)
	{
	}
	MasterPin(const MasterPin &m)
	{
		layer = m.layer;
		name = m.name;
	}
	void write(ostream &os);
};

struct Blockage
{
	string name;
	unsigned layer;
	int demand;
	Blockage(string n, unsigned l, int d) : name(n), layer(l), demand(d)
	{
	}
	Blockage(const Blockage &b)
	{
		name = b.name;
		layer = b.layer;
		demand = b.demand;
	}
	void write(ostream &os);
};

struct MasterCell
{
	string name;
	unsigned layers;
	unordered_map<string, MasterPin *> masterPinMap;
	vector<MasterPin *> masterPinList;
	unsigned numMasterPins;
	vector<Blockage> blockageList;
	MasterCell(string n, unsigned l, unsigned nmp, unordered_map<string, MasterPin *> &mPL,
			   const vector<Blockage> &bL, vector<MasterPin *> mpinl) : name(n), layers(l), numMasterPins(nmp), masterPinMap(mPL), blockageList(bL), masterPinList(mpinl)
	{
	}
	MasterPin *getMasterPin(string name)
	{
		return masterPinMap[name];
	}
	void write(ostream &os);
};

struct ExtraDemand
{
	MasterCell *cell1;
	MasterCell *cell2;
	Layer *layer;
	int demand;
	bool same; // true for same and false for adjacent
	ExtraDemand(MasterCell *c1, MasterCell *c2, Layer *l, int d, bool s) : cell1(c1), cell2(c2), layer(l), demand(d), same(s)
	{
	}
	void write(ostream &os);
};

struct Net;
struct Pin;

struct Inst
{
	string name;
	MasterCell *master;
	unsigned row;
	unsigned col;
	float cellCongestion;
	bool movable; // true if movable, false otherwise
	bool inFixedPartition;
	//unordered_map<string, Pin *> pinMap; //  We would also need a pin list since we now want to have a pin ID and a matrix related to that.
	vector<Pin *> instPinList;
	unordered_map<string, Pin *> pinMap;
	vector<Net *> netList;
	list<Inst *> adjacencyList;

	unsigned cellIndex;
	unsigned cellId;

	Inst(unsigned cid, string n, MasterCell *m, unsigned r, unsigned c, bool mov) : cellId(cid), name(n), master(m), row(r), col(c), movable(mov)
	{
		//pinMap.clear();
	}
	void write(ostream &os);
};

struct CompareCellCongestion
{
	bool operator()(Inst *const c1, Inst *const c2)
	{

		return ((c1->cellCongestion) < (c2->cellCongestion));
	}
};

struct Pin
{
	string name;
	Inst *inst;
	MasterPin *masterPin; // it refers to index  of the master pin
	Net *net;			  //refers to the Net which connects this pin
	vector<Pin *> pinAdjacencyList;
	// Here we have to add a pin ID (But this has to be assigned based upon current sliding window)
	unsigned pinIndexLocal;
	Pin()
	{
		//inst = NULL;
	}
	Pin(Inst *i, MasterPin *mp) : inst(i), masterPin(mp)
	{
	}
	Pin(string n, MasterPin *mp1) : name(n), masterPin(mp1)
	{
	}
	void write(ostream &os);
};

struct Route;

struct pinsForRouting
{
	unsigned row, col, layer;
	pinsForRouting(unsigned r, unsigned c, unsigned l) : row(r), col(c), layer(l)
	{
	}
};

struct Net
{
	string name;
	unsigned internalNetIndex = 0;
	unsigned pinSize;
	Layer *layer; // 0 when NoCstr, else valid min layer constraint
	vector<Pin *> pinList;
	list<Route *> segmentList;
	vector<pinsForRouting *> routerPinList; // For the net to be routed sorted pinlist with steiner points is traversed.
	bool inFixedPartition;
	/*
	 Net(Net &n)
	 {
	 name = n.name;
	 pinSize = n.pinSize;
	 layer = n.layer;
	 pinList = n.pinList;
	 //segmentList
	 }
	 */
	Net(string n, unsigned ps, Layer *l) : name(n), pinSize(ps), layer(l)
	{
	}
	void write(ostream &os);
	//unsigned int wireLength(void);
	unsigned hpwl(void);
};

struct Route
{
	unsigned sRow;
	unsigned sCol;
	unsigned sLayer;
	unsigned eRow;
	unsigned eCol;
	unsigned eLayer;
	//To store the direction (x, y or z) of the routed segment while parsing itself
	Direction axis;
	Net *net;
	Route(unsigned sR, unsigned sC, unsigned sL, unsigned eR, unsigned eC,
		  unsigned eL, Direction d, Net *n) : sRow(sR), sCol(sC), sLayer(sL), eRow(eR), eCol(eC), eLayer(eL), axis(d), net(n)
	{
	}
	void write(ostream &os);
};

struct routeGuide
{
	unsigned row;
	unsigned col;
	unsigned lay;
	unsigned pathCost;
	unsigned wireLengthCost;
	char pred;
	bool reached;
	bool expanded;
	routeGuide(unsigned R, unsigned C, unsigned L, unsigned pc, unsigned wc,
			   char p, bool r, bool e) : row(R), col(C), lay(L), pathCost(pc), wireLengthCost(wc), pred(p), reached(r), expanded(e)
	{
	}
	routeGuide(unsigned R, unsigned C, unsigned L, unsigned wc, char p, bool r,
			   bool e) : row(R), col(C), lay(L), wireLengthCost(wc), pred(p), reached(r), expanded(e)
	{
	}
};

struct CompareWirelengthCost
{
	bool operator()(routeGuide *const r1, routeGuide *const r2)
	{

		return r1->wireLengthCost > r2->wireLengthCost;
	}
};

struct CompareCellCost
{
	bool operator()(routeGuide *const r1, routeGuide *const r2)
	{

		return ((r1->pathCost) > (r2->pathCost));
	}
};

struct pinIndexes
{
	unsigned low;
	unsigned high;
	pinIndexes()
	{
	}
} /*pinIndexList[NUM_CELLS + 1]*/;

struct CellMove
{
	Inst *inst;
	unsigned newRow;
	unsigned newCol;
	CellMove(Inst *i, unsigned nR, unsigned nC) : inst(i), newRow(nR), newCol(nC)
	{
	}
	void write(ostream &os);
};

struct DesignData
{
	unsigned gGrid_demand[32][300][300];
	unsigned gGrid_supply[32][300][300];

	unsigned gGrid_demand_firstPartition[32][300][300];
	unsigned gGrid_demand_secondPartition[32][300][300];
	unsigned gGrid_demand_thirdPartition[32][300][300];
	unsigned gGrid_demand_fourthPartition[32][300][300];
	unsigned gGrid_demand_fifthPartition[32][300][300];
	unsigned gGrid_demand_sixthPartition[32][300][300];
	unsigned gGrid_demand_seventhPartition[32][300][300];
	unsigned gGrid_demand_eighthPartition[32][300][300];

	unsigned gGrid_supply_firstPartition[32][300][300];
	unsigned gGrid_supply_secondPartition[32][300][300];
	unsigned gGrid_supply_thirdPartition[32][300][300];
	unsigned gGrid_supply_fourthPartition[32][300][300];
	unsigned gGrid_supply_fifthPartition[32][300][300];
	unsigned gGrid_supply_sixthPartition[32][300][300];
	unsigned gGrid_supply_seventhPartition[32][300][300];
	unsigned gGrid_supply_eighthPartition[32][300][300];

	//float gGrid_congestion[32][300][300];
	routeGuide *gGrid_Route[32][300][300];

	vector<Layer *> layerList;
	vector<NonDefaultSupply *> ndsList;

	vector<MasterCell *> masterCellList;
	vector<ExtraDemand *> extraDemandList;
	vector<ExtraDemand *> extraDemandList_firstPartition;
	vector<ExtraDemand *> extraDemandList_secondPartition;
	vector<ExtraDemand *> extraDemandList_thirdPartition;
	vector<ExtraDemand *> extraDemandList_fourthPartition;
	vector<ExtraDemand *> extraDemandList_fifthPartition;
	vector<ExtraDemand *> extraDemandList_sixthPartition;
	vector<ExtraDemand *> extraDemandList_seventhPartition;
	vector<ExtraDemand *> extraDemandList_eighthPartition;

	vector<Net *> netList;
	vector<Net *> fixedNetList;
	vector<Net *> firstPartitionNetList;
	vector<Net *> secondPartitionNetList;
	vector<Net *> thirdPartitionNetList;
	vector<Net *> fourthPartitionNetList;
	vector<Net *> fifthPartitionNetList;
	vector<Net *> sixthPartitionNetList;
	vector<Net *> seventhPartitionNetList;
	vector<Net *> eighthPartitionNetList;

	vector<Inst *> instList;
	vector<Inst *> instListFirstPartition;
	vector<Inst *> instListSecondPartition;
	vector<Inst *> instListThirdPartition;
	vector<Inst *> instListFourthPartition;
	vector<Inst *> instListFifthPartition;
	vector<Inst *> instListSixthPartition;
	vector<Inst *> instListSeventhPartition;
	vector<Inst *> instListEighthPartition;
	vector<Inst *> instListFixedPartition;

	vector<Route *> routeList;
	vector<Route *> routeList_firstPartition;
	vector<Route *> routeList_secondPartition;
	vector<Route *> routeList_thirdPartition;
	vector<Route *> routeList_fourthPartition;
	vector<Route *> routeList_fifthPartition;
	vector<Route *> routeList_sixthPartition;
	vector<Route *> routeList_seventhPartition;
	vector<Route *> routeList_eighthPartition;

	vector<CellMove *> cmList;
	vector<CellMove *> cmList_firstPartition;
	vector<CellMove *> cmList_secondPartition;
	vector<CellMove *> cmList_thirdPartition;
	vector<CellMove *> cmList_fourthPartition;
	vector<CellMove *> cmList_fifthPartition;
	vector<CellMove *> cmList_sixthPartition;
	vector<CellMove *> cmList_seventhPartition;
	vector<CellMove *> cmList_eighthPartition;

	unordered_map<string, NonDefaultSupply *> ndsMap;
	unordered_map<string, Layer *> layerMap;
	unordered_map<string, MasterCell *> mcMap;
	unordered_map<string, Inst *> instMap;
	unordered_map<string, Net *> netMap;

	unordered_map<string, Supply *> supplyMap;
	//unordered_map<string, vector<Inst *> > instAdjacencyList;

	//priority_queue<routeGuide*, vector<routeGuide*>, CompareWirelengthCost> wavefront;
	//queue<routeGuide *> q;

	void write(ostream &os);
	Layer *getLayerFromName(string name);
	MasterCell *getMasterCellFromName(string name);
	Inst *getInstanceFromName(string name);
	Inst *getInstanceFromCellIndex(unsigned cid);
	Pin *getPinFromName(string name); // name is instance_name/pin_name
	Net *getNetFromName(string name);
	unsigned int totalHpwl(ofstream &os);
	void produceOutput(string name);
	void setDemandtoDemandMatrix(int demVal, unsigned lay, unsigned row, unsigned col)
	{
		gGrid_demand[lay][row][col] = demVal;
	}
};

class GlobalVariables
{
public:
	unsigned maxCellMove;				 // Ignore this for now
	unsigned rowBeginIdx;				 // -> Work on this
	unsigned colBeginIdx;				 // -> Work on this
	unsigned rowEndIdx;					 // -> Work on this
	unsigned colEndIdx;					 // -> Work on this
	unsigned numLayers;					 // ignore this
	unsigned numNonDefaultSupplyGGrid;	 // ignore this
	unsigned masterCellCount;			 // ignore this
	unsigned numNeighborCellExtraDemand; // ignore this
	unsigned numCellInst;				 // ignore this
	unsigned numNets;					 // ignore this
	unsigned numRoutes;					 // ignore this
	//unsigned pinIdxCounter = 0;

	//ML data requirements

	unsigned n = NUM_ROWS, m = NUM_COLS, l = NUM_LAYERS, k = NUM_CELLS /*Need to move k to local memory*/, a = 1;
	bool connectivity_matrix[2001][2001];																							// Done
	bool pin_connectivity_matrix[2001][2001];																						// Remove
	bool congestion_matrix[32][300][300];																							// Not needed in current algorithm
	unsigned rowPosCell[(NUM_CELLS + 1)], colPosCell[(NUM_CELLS + 1)], layPosCell[(NUM_CELLS + 1)], demandValCell[(NUM_CELLS + 1)]; // Remove
	unsigned cellIdxList[NUM_CELLS + 1], movableCellIdxList[NUM_CELLS + 1];															// Remove
	std::unordered_set<string> pinNetList;																							// safely taking double the size. -->Remove
	struct pinIndexes																												//remove
	{
		unsigned low;
		unsigned high;
		pinIndexes()
		{
		}
	} pinIndexList[NUM_CELLS + 1];
	unsigned numPinsArray[NUM_CELLS + 1]; // Remove
	list<Inst *> cellsInRowWindow;		  // Remove
	list<Inst *> cellsBeyondRowWindow;	  // Remove
	//list <Inst*> cellsInColWindow;
	//list <Inst*> cellsBeyondColWindow;

	list<Inst *> cellsInWindow; // Remove

	list<Inst *> movableCellList; // Remove

	float X[149]; // Remove
	//float pred[149];

	float Y[2]; // Remove
	unsigned numcellsmoved;

	DesignData dd;
};
