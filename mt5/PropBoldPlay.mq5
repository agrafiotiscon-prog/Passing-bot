//+------------------------------------------------------------------+
//|                                                 PropBoldPlay.mq5 |
//|  Prop-firm challenge manager: "bold play" sizing + rule guards.   |
//|                                                                  |
//|  What it does                                                    |
//|   * One market entry per day at a fixed server time.             |
//|   * Stop = InpStopAtr x daily ATR(InpAtrDays).                    |
//|   * Take-profit placed so that a win lifts equity to the          |
//|     challenge target (+10%) in one trade ("bold play").          |
//|   * Risk per trade = min(InpRiskPct, room to the daily halt,      |
//|     room to the overall floor).                                  |
//|   * Guards on every tick/second: target lock (close all, stop),   |
//|     daily equity halt, overall floor, challenge timer.           |
//|                                                                  |
//|  Honest expectation (see README / research/final_eval.py):       |
//|   ~35-40% of 14-day challenges pass. This comes from optimal     |
//|   sizing for the target/limit structure, NOT from a market edge; |
//|   it is not a profitable strategy for a funded account.          |
//+------------------------------------------------------------------+
#property copyright "Passing-bot"
#property version   "1.01"
#property description "Prop challenge bold-play manager. ~35-40% pass rate per attempt in backtests 2012-2026; no long-run edge."

#include <Trade\Trade.mqh>

enum ENUM_DIR_MODE
  {
   DIR_TREND = 0,   // sign of the last N daily closes
   DIR_COIN  = 1,   // coin flip
   DIR_LONG  = 2,
   DIR_SHORT = 3
  };

input group "Challenge rules (as set by the firm)"
input double InpInitialBalance   = 0.0;    // Initial balance (0 = balance when the EA first runs)
input double InpTargetPct        = 10.0;   // Profit target, % of initial
input double InpDailyLossPct     = 3.0;    // Max daily loss, % of initial
input double InpMaxLossPct       = 6.0;    // Max overall loss, % of initial
input int    InpChallengeDays    = 14;     // Time limit, calendar days
input int    InpDayResetHour     = 0;      // Server hour when the firm's trading day (daily-loss reset) starts
input bool   InpResetState       = false;  // Reset stored challenge state on start

input group "Risk management"
input double InpRiskPct          = 2.9;    // Max risk per trade, % of initial
input double InpDailyHaltPct     = 2.9;    // Flatten & stop for the day at this daily loss, %
input double InpFloorBufferPct   = 0.2;    // Keep this distance from the overall floor, %
input double InpTargetBufferPct  = 0.05;   // Aim this far above the target, %
input double InpMaxRR            = 20.0;   // Cap on take-profit distance, x stop distance

input group "Entry / exit (server time)"
input int    InpEntryHour        = 10;
input int    InpEntryMinute      = 0;
input int    InpExitHour         = 23;     // flat before the daily rollover
input int    InpExitMinute       = 45;
input bool   InpTradeWeekends    = false;  // true for 24/7 crypto CFDs
input ENUM_DIR_MODE InpDirMode   = DIR_TREND;
input int    InpTrendDays        = 10;
input int    InpAtrDays          = 14;
input double InpStopAtr          = 0.2;    // stop distance, x daily ATR

input group "Execution"
input ulong  InpMagic            = 20260928;
input int    InpDeviationPoints  = 50;

CTrade  g_trade;
string  g_pfx;
double  g_init;
datetime g_start;

//--- persistent state helpers (terminal global variables) ---------------
string Key(const string name) { return g_pfx + name; }
double GetState(const string name, const double def)
  {
   return GlobalVariableCheck(Key(name)) ? GlobalVariableGet(Key(name)) : def;
  }
void SetState(const string name, const double v) { GlobalVariableSet(Key(name), v); }

long ServerDay(const datetime t) { return (long)(t / 86400); }

//+------------------------------------------------------------------+
int OnInit()
  {
   g_pfx = StringFormat("PBP_%I64d_%I64u_%s_", AccountInfoInteger(ACCOUNT_LOGIN), InpMagic, _Symbol);
   if(InpResetState)
     {
      string names[] = {"init", "start", "dayid", "dayref", "halted", "done", "lastday", "missday", "tryday", "tries"};
      for(int i = 0; i < ArraySize(names); i++)
         GlobalVariableDel(Key(names[i]));
     }
   if(!GlobalVariableCheck(Key("init")))
     {
      double b = InpInitialBalance > 0 ? InpInitialBalance : AccountInfoDouble(ACCOUNT_BALANCE);
      SetState("init", b);
      SetState("start", (double)TimeCurrent());
      SetState("done", 0);
     }
   g_init  = GetState("init", AccountInfoDouble(ACCOUNT_BALANCE));
   g_start = (datetime)GetState("start", (double)TimeCurrent());
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpDeviationPoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   MathSrand((uint)GetTickCount());
   EventSetTimer(1);
   double pre[];
   int got = CopyClose(_Symbol, PERIOD_D1, 0, InpAtrDays * 10 + InpTrendDays + 2, pre);  // start loading D1 history now
   PrintFormat("PropBoldPlay: initial=%.2f start=%s target=%.2f floor=%.2f",
               g_init, TimeToString(g_start), g_init * (1 + InpTargetPct / 100), g_init * (1 - InpMaxLossPct / 100));
   PrintFormat("PropBoldPlay: server time %s, daily entry at %02d:%02d server time, D1 bars loaded: %d",
               TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES), InpEntryHour, InpEntryMinute, got);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) { EventKillTimer(); Comment(""); }

//+------------------------------------------------------------------+
int CountPositions()
  {
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == (long)InpMagic && PositionGetString(POSITION_SYMBOL) == _Symbol)
         n++;
     }
   return n;
  }

void CloseAll(const string why)
  {
   for(int attempt = 0; attempt < 5; attempt++)
     {
      bool left = false;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong tk = PositionGetTicket(i);
         if(tk == 0) continue;
         if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagic || PositionGetString(POSITION_SYMBOL) != _Symbol)
            continue;
         if(!g_trade.PositionClose(tk))
            left = true;
        }
      if(!left) break;
      Sleep(200);
     }
   if(why != "") Print("CloseAll: ", why);
  }

//--- Wilder ATR of completed daily bars (matches research/strategies.py) --
double DailyATR()
  {
   int need = InpAtrDays * 10 + 2;
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int got = CopyRates(_Symbol, PERIOD_D1, 1, need, r);   // oldest..newest, excludes today
   if(got < InpAtrDays + 2) return 0.0;
   double atr = r[1].high - r[1].low;
   double a = 1.0 / InpAtrDays;
   for(int i = 1; i < got; i++)
     {
      double pc = r[i - 1].close;
      double tr = MathMax(r[i].high - r[i].low, MathMax(MathAbs(r[i].high - pc), MathAbs(r[i].low - pc)));
      atr = a * tr + (1 - a) * atr;
     }
   return atr;
  }

int Direction()
  {
   if(InpDirMode == DIR_LONG)  return 1;
   if(InpDirMode == DIR_SHORT) return -1;
   if(InpDirMode == DIR_COIN)  return (MathRand() % 2 == 0) ? 1 : -1;
   double c[];
   ArraySetAsSeries(c, false);
   int need = InpTrendDays + 1;
   if(CopyClose(_Symbol, PERIOD_D1, 1, need, c) < need) return 0;   // D1 history not ready yet
   double c1 = c[need - 1];   // yesterday's close
   double cn = c[0];          // close InpTrendDays days before that
   if(c1 <= 0 || cn <= 0 || c1 == cn) return 0;
   return c1 > cn ? 1 : -1;
  }

//--- make sure daily history is downloaded before the entry window ----------
bool DailyHistoryReady()
  {
   double c[];
   int need = InpAtrDays * 10 + InpTrendDays + 2;
   return CopyClose(_Symbol, PERIOD_D1, 1, need, c) >= InpAtrDays + InpTrendDays + 2;
  }

//--- throttled "why no trade" logging --------------------------------------
string   g_note = "";
datetime g_note_time = 0;
void Note(const string why)
  {
   if(why != g_note || TimeCurrent() - g_note_time >= 300)
     {
      Print("PropBoldPlay: no entry yet - ", why);
      g_note_time = TimeCurrent();
     }
   g_note = why;
  }

//--- money value of a 1.0-lot price move of `dist` ------------------------
double ValuePerLot(const double dist, const bool loss)
  {
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tv = SymbolInfoDouble(_Symbol, loss ? SYMBOL_TRADE_TICK_VALUE_LOSS : SYMBOL_TRADE_TICK_VALUE_PROFIT);
   if(tv <= 0) tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(ts <= 0 || tv <= 0) return 0.0;
   return dist / ts * tv;
  }

double NormalizeLots(double lots)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0) step = 0.01;
   lots = MathFloor(lots / step) * step;
   if(lots > vmax) lots = vmax;
   if(lots < vmin) return 0.0;
   return NormalizeDouble(lots, 8);
  }

//+------------------------------------------------------------------+
void Manage()
  {
   datetime now = TimeCurrent();
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double done = GetState("done", 0);

   double target_lvl = g_init * (1.0 + (InpTargetPct + InpTargetBufferPct) / 100.0);
   double floor_lvl  = g_init * (1.0 - InpMaxLossPct / 100.0);

   //--- new firm day: daily reference = max(balance, equity) at the reset time
   long day = ServerDay(now - (datetime)InpDayResetHour * 3600);
   if((long)GetState("dayid", -1) != day)
     {
      SetState("dayid", (double)day);
      SetState("dayref", MathMax(bal, eq));
      SetState("halted", 0);
     }
   double dayref   = GetState("dayref", MathMax(bal, eq));
   double halt_lvl = dayref - InpDailyHaltPct / 100.0 * g_init;
   bool   halted   = GetState("halted", 0) > 0;

   string status = done == 1 ? "PASSED" : done == -1 ? "STOPPED (floor)" : done == 2 ? "TIME OVER" : halted ? "halted today" : "active";
   string traded = (long)GetState("lastday", -1) == day ? "traded today" : StringFormat("entry %02d:%02d server", InpEntryHour, InpEntryMinute);
   Comment(StringFormat("PropBoldPlay  %s\nequity %.2f  target %.2f  floor %.2f\nday ref %.2f  halt %.2f  day %d/%d\n%s  (server %s)%s",
                        status, eq, target_lvl, floor_lvl, dayref, halt_lvl,
                        (int)((now - g_start) / 86400) + 1, InpChallengeDays,
                        traded, TimeToString(now, TIME_MINUTES), g_note == "" ? "" : "\nwaiting: " + g_note));

   if(done != 0)
     {
      if(CountPositions() > 0) CloseAll("challenge finished");
      return;
     }
   //--- guards
   if(eq >= target_lvl)
     {
      CloseAll("target reached");
      SetState("done", 1);
      Print("PropBoldPlay: TARGET REACHED, trading stopped. Equity ", DoubleToString(eq, 2));
      return;
     }
   if(eq <= floor_lvl + InpFloorBufferPct / 100.0 * g_init * 0.5)
     {
      CloseAll("overall floor");
      SetState("done", -1);
      return;
     }
   if(now >= g_start + (datetime)InpChallengeDays * 86400)
     {
      CloseAll("time limit");
      SetState("done", 2);
      return;
     }
   if(!halted && eq <= halt_lvl)
     {
      CloseAll("daily halt");
      SetState("halted", 1);
      halted = true;
     }

   MqlDateTime t;
   TimeToStruct(now, t);
   int mins = t.hour * 60 + t.min;
   //--- time exit
   if(mins >= InpExitHour * 60 + InpExitMinute && CountPositions() > 0)
      CloseAll("time exit");

   //--- entry
   if(halted || CountPositions() > 0) return;
   if(!InpTradeWeekends && (t.day_of_week == 0 || t.day_of_week == 6)) return;
   int e0 = InpEntryHour * 60 + InpEntryMinute;
   if((long)GetState("lastday", -1) == day) return;             // already traded today
   if(mins < e0) return;
   if(mins >= e0 + 60 || mins >= InpExitHour * 60 + InpExitMinute)
     {
      // window over without a trade: say why once, then wait for tomorrow
      if((long)GetState("missday", -1) != day)
        {
         SetState("missday", (double)day);
         Print("PropBoldPlay: entry window closed with no trade today. Last reason: ",
               g_note == "" ? "EA was not running during the window" : g_note);
        }
      return;
     }
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     { Note("Algo Trading is switched off (toolbar button or EA Common tab)"); return; }
   if(!DailyHistoryReady())
     { Note("daily (D1) history still loading - retrying"); return; }

   int dir = Direction();
   if(dir == 0) { Note("trend direction undefined (D1 closes equal or not loaded) - retrying"); return; }
   double atr = DailyATR();
   if(atr <= 0) { Note("daily ATR not available yet - retrying"); return; }
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double px  = dir > 0 ? ask : bid;
   double sl_dist = InpStopAtr * atr;
   double min_dist = (SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) + 2) * _Point;
   if(sl_dist < min_dist) sl_dist = min_dist;
   double sl = NormalizeDouble(px - dir * sl_dist, _Digits);
   double spread = ask - bid;

   //--- risk budget (money)
   double risk_amt = InpRiskPct / 100.0 * g_init;
   risk_amt = MathMin(risk_amt, eq - halt_lvl);
   risk_amt = MathMin(risk_amt, eq - floor_lvl - InpFloorBufferPct / 100.0 * g_init);
   if(risk_amt <= 0)
     {
      SetState("lastday", (double)day);
      Print("PropBoldPlay: no risk budget left (too close to the daily halt or overall floor) - no trade");
      return;
     }
   double loss_per_lot = ValuePerLot(MathAbs(px - sl) + spread, true);
   if(loss_per_lot <= 0) { Note("symbol tick value not available yet - retrying"); return; }
   double lots = NormalizeLots(risk_amt / loss_per_lot);
   if(lots <= 0)
     {
      SetState("lastday", (double)day);
      Print("PropBoldPlay: lot size below the broker minimum for this risk - no trade today");
      return;
     }
   //--- margin check
   double margin = 0;
   ENUM_ORDER_TYPE ot = dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   while(lots > 0 && OrderCalcMargin(ot, _Symbol, lots, px, margin) && margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.95)
      lots = NormalizeLots(lots * 0.9);
   if(lots <= 0)
     {
      SetState("lastday", (double)day);
      Print("PropBoldPlay: not enough free margin even for the minimum lot - check account leverage");
      return;
     }

   //--- bold take-profit: a win lifts equity to the target
   double need = target_lvl - eq;
   double gain_per_lot_unit = ValuePerLot(1.0, false);           // money per 1.0 price move per lot
   double tp_dist = need / (lots * gain_per_lot_unit);
   if(dir < 0) tp_dist += spread;                                   // short TP triggers on the ask
   if(tp_dist > InpMaxRR * sl_dist) tp_dist = InpMaxRR * sl_dist;
   double tp = NormalizeDouble(px + dir * tp_dist, _Digits);

   bool ok = dir > 0 ? g_trade.Buy(lots, _Symbol, 0.0, sl, tp, "PBP")
                     : g_trade.Sell(lots, _Symbol, 0.0, sl, tp, "PBP");
   ok = ok && (g_trade.ResultRetcode() == TRADE_RETCODE_DONE || g_trade.ResultRetcode() == TRADE_RETCODE_PLACED);
   PrintFormat("PropBoldPlay: %s %.2f lots @%.5f SL %.5f TP %.5f risk %.2f need %.2f -> %s",
               dir > 0 ? "BUY" : "SELL", lots, px, sl, tp, risk_amt, need, ok ? "ok" : g_trade.ResultRetcodeDescription());
   if(ok)
     {
      SetState("lastday", (double)day);       // one trade per day
      g_note = "";
      return;
     }
   // failed send (requote, off quotes, ...): retry a few times inside the window
   double tries = (long)GetState("tryday", -1) == day ? GetState("tries", 0) + 1 : 1;
   SetState("tryday", (double)day);
   SetState("tries", tries);
   if(tries >= 5)
     {
      SetState("lastday", (double)day);
      Print("PropBoldPlay: order rejected 5 times today - giving up until tomorrow");
     }
   else
      Note("order rejected (" + g_trade.ResultRetcodeDescription() + ") - retrying");
  }

void OnTick()  { Manage(); }
void OnTimer() { Manage(); }
//+------------------------------------------------------------------+
