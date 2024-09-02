using log4net;
using log4net.Appender;
using log4net.Repository.Hierarchy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;

namespace FishingFun
{
    public class FishingBotAI
    {
        public static ILog logger = LogManager.GetLogger("Fishbot");

        private IBobberFinder bobberFinder;
        private IBiteWatcher biteWatcher;
        private ConsoleKey castKey;
        private List<ConsoleKey> tenMinKey;
        private bool isEnabled;
        private Stopwatch stopwatch = new Stopwatch();
        private static Random random = new Random();
        public event EventHandler<FishingEvent> FishingEventHandler;


        public FishingBotAI(IBobberFinder bobberFinder, IBiteWatcher biteWatcher, ConsoleKey castKey, List<ConsoleKey> tenMinKey)
        {
            this.bobberFinder = bobberFinder;
            this.biteWatcher = biteWatcher;
            this.castKey = castKey;
            this.tenMinKey = tenMinKey;

            logger.Info("FishBotAI Created.");

            FishingEventHandler += (s, e) => { };
        }

        public void Start()
        {
            biteWatcher.FishingEventHandler = (e) => FishingEventHandler?.Invoke(this, e);

            isEnabled = true;

            DoTenMinuteKey();

            ScreenRecorder.SetBufferSize(Model.SEQUENCE_LENGTH);

            while (isEnabled)
            {
                try
                {
                    logger.Info($"Pressing key {castKey} to Cast.");

                    PressTenMinKeyIfDue();

                    FishingEventHandler?.Invoke(this, new FishingEvent { Action = FishingAction.Cast });

                    WowProcess.PressKey(castKey);
                       
                    ScreenRecorder.StartLoop();

                    Watch(2000); 

                    var bobberPosition = WaitForBite();

                    Thread.Sleep(150); // If recording

                    ScreenRecorder.StopLoop();

                    if (bobberPosition != Point.Empty)
                    {
                        Loot(bobberPosition);
                    }
                    else
                    {
                        logger.Info("failed to loot");
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


        public void Stop()
        {
            isEnabled = false;
            logger.Error("Bot is Stopping...");
        }

        private Point WaitForBite()
        {
            bobberFinder.Reset();

            var bobberPosition = FindBobber();
            if (bobberPosition == Point.Empty)
            {
                return Point.Empty;
            }

            this.biteWatcher.Reset(bobberPosition);

            logger.Info("Bobber start position: " + bobberPosition);

            var timedTask = new TimedAction((a) => { logger.Info("Fishing timed out!"); }, 25 * 1000, 25);

            // Wait for the bobber to move
            while (isEnabled)
            {
                var currentBobberPosition = FindBobber();
                if (currentBobberPosition == Point.Empty) { return Point.Empty; }

                if (this.biteWatcher.IsBite(currentBobberPosition))
                {
                    return bobberPosition;
                }

                if (!timedTask.ExecuteIfDue()) { return Point.Empty; }
            }

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