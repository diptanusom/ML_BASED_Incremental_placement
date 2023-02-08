#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <assert.h>

#include "data.hxx"

void Layer::write(ostream &os)
{
   os << "Lay " << name << " " << number << " " << ((horizontal) ? "H" : "V") << " " << default_supply << endl;
}

void NonDefaultSupply::write(ostream &os)
{
   os << row << " " << column << " " << layer << " ";
   if (value >= 0)
      os << "+";
   os << value << endl;
}

void MasterPin::write(ostream &os)
{
   os << "Pin " << name << " M" << layer << endl;
}

void Blockage::write(ostream &os)
{
   os << "Blkg " << name << " M" << layer << " " << demand << endl;
}

void ExtraDemand::write(ostream &os)
{
   if (same)
      os << "sameGGrid ";
   else
      os << "adjHGGrid ";
   os << cell1->name << " " << cell2->name << " " << layer->name << " " << demand << endl;
}

void MasterCell::write(ostream &os)
{
   os << "MasterCell " << name << " " << masterPinMap.size() << " " << blockageList.size() << endl;
   for (unordered_map<string, MasterPin *>::iterator i = masterPinMap.begin(); i != masterPinMap.end(); ++i)
   {
      (i->second)->write(os);
   }

   for (vector<Blockage>::iterator i = blockageList.begin(); i != blockageList.end(); ++i)
   {
      (*i).write(os);
   }
}

void Inst::write(ostream &os)
{
   os << "CellInst " << name << " " << master->name << " " << row << " " << col << " ";
   if (movable)
      os << "Movable" << endl;
   else
      os << "Fixed" << endl;
}

void Pin::write(ostream &os)
{
   os << "Pin " << inst->name << "/" << masterPin->name << endl;
}

void Net::write(ostream &os)
{
   os << "Net " << name << " " << pinList.size() << " ";
   if (layer)
      os << layer->name << endl;
   else
      os << "NoCstr" << endl;
   for (vector<Pin *>::iterator i = pinList.begin(); i != pinList.end(); ++i)
   {
      (*i)->write(os);
   }
}

unsigned
Net::hpwl(void) // TODO : Fix the HPWL 
{
   unsigned hpwirelength;
   vector<Pin *>::iterator p = pinList.begin(); // Pinlist inside teh particular net
   unsigned xmin, xmax, ymin, ymax;
   unsigned x, y;

   xmin = xmax = (*p)->inst->row;
   ymin = ymax = (*p)->inst->col;
   ++p;

   for (; p != pinList.end(); ++p)
   {
      x = (*p)->inst->row;
      y = (*p)->inst->col;

      if (x < xmin)
      {
         xmin = x;
      }
      else if (x > xmax)
      {
         xmax = x;
      }

      if (y < ymin)
      {
         ymin = y;
      }
      else if (y > ymax)
      {
         ymax = y;
      }
   }

   hpwirelength = (xmax - xmin) + (ymax - ymin);

   return hpwirelength;
}

void Route::write(ostream &os)
{
   os << sRow << " " << sCol << " " << sLayer << " " << eRow << " " << eCol << " " << eLayer << " " << net->name << endl;
}

void CellMove::write(ostream &os)
{
   os << "CellInst"
      << " " << inst->name << " " << newRow << " " << newCol << endl;
}

void DesignData::write(ostream &os)
{
   for (vector<Layer *>::iterator i = layerList.begin(); i != layerList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumNonDefaultSupplyGGrid " << ndsList.size() << endl;

   for (vector<NonDefaultSupply *>::iterator i = ndsList.begin(); i != ndsList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumMasterCell " << masterCellList.size() << endl;

   for (vector<MasterCell *>::iterator i = masterCellList.begin(); i != masterCellList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumNeighborCellExtraDemand " << extraDemandList.size() << endl;

   for (vector<ExtraDemand *>::iterator i = extraDemandList.begin(); i != extraDemandList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumCellInst " << instList.size() << endl;

   for (vector<Inst *>::iterator i = instList.begin(); i != instList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumNets " << netList.size() << endl;

   for (vector<Net *>::iterator i = netList.begin(); i != netList.end(); ++i)
   {
      (*i)->write(os);
   }

   os << "NumRoutes " << routeList.size() << endl;

   for (vector<Route *>::iterator i = routeList.begin(); i != routeList.end(); ++i)
   {
      (*i)->write(os);
   }
}

void DesignData::produceOutput(string name)
{
   ofstream file;
   file.open(name.c_str());
   file << "NumMovedCellInst " << cmList.size() << endl;

   for (vector<CellMove *>::iterator i = cmList.begin(); i != cmList.end(); ++i)
   {
      (*i)->write(file);
   }

   file << "NumRoutes " << routeList.size() << endl;

   for (vector<Route *>::iterator i = routeList.begin(); i != routeList.end(); ++i)
   {
      (*i)->write(file);
   }

   file.close();
}

Layer *
DesignData::getLayerFromName(string name)
{
   return layerMap[name];
}

MasterCell *
DesignData::getMasterCellFromName(string name)
{
   return mcMap[name];
}

Inst *
DesignData::getInstanceFromName(string name)
{
   return instMap[name];
}

Inst *
DesignData::getInstanceFromCellIndex(unsigned cid)
{
   //cout << endl << "Trying to find cell with ID : " << cid;
   //exit(0);
   for (vector<Inst *>::iterator itr = instList.begin(); itr != instList.end(); itr++)
   {
      if ((*itr)->cellIndex == cid)
         return (*itr);
   }
}

Pin *DesignData::getPinFromName(string name)
{
   stringstream ss(name); // Turn the string into a stream.
   string tok;
   getline(ss, tok, '/');
   Inst *inst = getInstanceFromName(tok);
   assert(inst);

   // Find if pin is already created
   getline(ss, tok, '/');
   string pinName = tok;
   unordered_map<string, Pin *>::iterator itr = inst->pinMap.find(pinName);

   if (itr == inst->pinMap.end())
   { // did not find, so create it
      MasterPin *mp = inst->master->getMasterPin(pinName);
      assert(mp);
      Pin *pin = new Pin(inst, mp);
      pin->name = pinName;
      inst->pinMap[pinName] = pin;
      pin->pinIndexLocal = atoi(tok.c_str() + 1);
      return pin;
   }

   // found it
   return itr->second;
}

Net *DesignData::getNetFromName(string name)
{
   return netMap[name];
}

unsigned int
DesignData::totalHpwl(ofstream &logs)
{
   unsigned int totalhpwl = 0, hpwirelength = 0;

   for (vector<Net *>::iterator i = netList.begin(); i != netList.end(); ++i)
   {
      hpwirelength = (*i)->hpwl(); // ! Calculating wirelength of one net in the netlist
      totalhpwl += hpwirelength; // ! adding hpwirelength for temporary case

      // << (*i)->name << " HPWL: " << hpwirelength << endl;
   }
   return totalhpwl; //returning total hpwl for use in simulated annealing results comparison
}
