using log4net;
using log4net.Appender;
using log4net.Repository.Hierarchy;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Threading;

namespace FishingFun
{
    public class FishingBot
    {
        public static ILog logger = LogManager.GetLogger("Fishbot");

        private ConsoleKey castKey;
        private List<ConsoleKey> tenMinKey;
        private IBobberFinder bobberFinder;
        private IBiteWatcher biteWatcher;
        private bool isEnabled;
        private (int, int) mouseClicked = (0, 0);
        private Stopwatch stopwatch = new Stopwatch();
        private static Random random = new Random();
        private static readonly string savePath = "E://fishdata";

        public event EventHandler<FishingEvent> FishingEventHandler;


        public FishingBot(IBobberFinder bobberFinder, IBiteWatcher biteWatcher, ConsoleKey castKey, List<ConsoleKey> tenMinKey)
        {
            this.bobberFinder = bobberFinder;
            this.biteWatcher = biteWatcher;
            this.castKey = castKey;
            this.tenMinKey = tenMinKey;

            logger.Info("FishBot Created.");

            FishingEventHandler += (s, e) => { };

        }

        public void Start()
        {
            biteWatcher.FishingEventHandler = (e) => FishingEventHandler?.Invoke(this, e);

            isEnabled = true;

            DoTenMinuteKey();

            ScreenRecorder.SetBufferSize(40);

            while (isEnabled)
            {
                try
                {
                    logger.Info($"Pressing key {castKey} to Cast.");

                    PressTenMinKeyIfDue();

                    FishingEventHandler?.Invoke(this, new FishingEvent { Action = FishingAction.Cast });

                    Thread.Sleep(250);

                    WowProcess.PressKey(castKey);
                       
                    ScreenRecorder.StartLoop();

                    Watch(2000);

                    var bobberPosition = WaitForBite();

                    Thread.Sleep(200);  // last_n_pos += 1 for every 200ms

                    ScreenRecorder.StopLoop();

                    if (bobberPosition != Point.Empty)
                    {
                        Loot(bobberPosition);
                        ScreenRecorder.SaveRecords(WowScreen.GetBitmapPositionFromScreenPosition(bobberPosition)， savePath);
                    }

                    ScreenRecorder.EnsureEmpty();

                }
                catch (Exception e)
                {
                    logger.Error(e.ToString());
                    Sleep(2000);
                }
            }

            logger.Error("Bot has Stopped.");
        }


        public void SetCastKey(ConsoleKey castKey)
        {
            this.castKey = castKey;
        }

        private void Watch(int milliseconds)
        {
            bobberFinder.Reset();
            stopwatch.Reset();
            stopwatch.Start();
            while (stopwatch.ElapsedMilliseconds < milliseconds)
            {
                bobberFinder.Find();
            }
            stopwatch.Stop();
        }

        public void Stop()
        {
            isEnabled = false;
            logger.Error("Bot is Stopping...");
        }

        private Point WaitForBite()
        {
            bobberFinder.Reset();
            mouseClicked = (0, 0);

            var bobberPosition = FindBobber();
            if (bobberPosition == Point.Empty)
            {
                return Point.Empty;
            }

            this.biteWatcher.Reset(bobberPosition);

            logger.Info("Bobber start position: " + bobberPosition);

            var timedTask = new TimedAction((a) => { logger.Info("Fishing timed out!"); }, 25 * 1000, 25);

            // Wait for the bobber to move
            while (isEnabled && mouseClicked == (0, 0))
            {
                var currentBobberPosition = FindBobber();
                if (currentBobberPosition == Point.Empty || currentBobberPosition.X == 0) { return Point.Empty; }

                if (this.biteWatcher.IsBite(currentBobberPosition))
                {
                    return bobberPosition;
                }

                if (!timedTask.ExecuteIfDue()) { return Point.Empty; }
            }

            if (mouseClicked != (0, 0))
                return new Point(mouseClicked.Item1, mouseClicked.Item2);

            return Point.Empty;
        }

        private DateTime StartTime = DateTime.Now;

        private void PressTenMinKeyIfDue()
        {
            if ((DateTime.Now - StartTime).TotalMinutes > 10 && tenMinKey.Count > 0)
            {
                DoTenMinuteKey();
            }
        }

        /// <summary>
        /// Ten minute key can do anything you want e.g.
        /// Macro to apply a lure: 
        /// /use Bright Baubles
        /// /use 16
        /// 
        /// Or a macro to delete junk:
        /// /run for b=0,4 do for s=1,GetContainerNumSlots(b) do local n=GetContainerItemLink(b,s) if n and (strfind(n,"Raw R") or strfind(n,"Raw Spot") or strfind(n,"Raw Glo") or strfind(n,"roup")) then PickupContainerItem(b,s) DeleteCursorItem() end end end
        /// </summary>
        private void DoTenMinuteKey()
        {
            StartTime = DateTime.Now;

            if (tenMinKey.Count == 0)
            {
                logger.Info($"Ten Minute Key:  No keys defined in tenMinKey, so nothing to do (Define in call to FishingBot constructor).");
            }

            FishingEventHandler?.Invoke(this, new FishingEvent { Action = FishingAction.Cast });

            foreach (var key in tenMinKey)
            {
                logger.Info($"Ten Minute Key: Pressing key {key} to run a macro, delete junk fish or apply a lure etc.");
                WowProcess.PressKey(key);
            }
        }

        private void Loot(Point bobberPosition)
        {
            logger.Info($"Right clicking mouse to Loot.");
            WowProcess.RightClickMouse(logger, bobberPosition);
        }

        public static void Sleep(int ms)
        {
            ms+=random.Next(0, 225);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            while (sw.Elapsed.TotalMilliseconds < ms)
            {
                FlushBuffers();
                Thread.Sleep(100);
            }
        }

        public static void FlushBuffers()
        {
            ILog log = LogManager.GetLogger("Fishbot");
            var logger = log.Logger as Logger;
            if (logger != null)
            {
                foreach (IAppender appender in logger.Appenders)
                {
                    var buffered = appender as BufferingAppenderSkeleton;
                    if (buffered != null)
                    {
                        buffered.Flush();
                    }
                }
            }
        }

        private Point FindBobber()
        {
            var timer = new TimedAction((a) => { logger.Info("Waited seconds for target: " + a.ElapsedSecs); }, 1000, 5);

            while (true)
            {
                var target = this.bobberFinder.Find();
                if (target != Point.Empty || !timer.ExecuteIfDue()) { return target; }
            }
        }
    }
}