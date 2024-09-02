using log4net;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading;
using System.Windows.Forms;

namespace FishingFun
{
    public static class WowScreen
    {
        private static ILog logger = LogManager.GetLogger("Fishbot");
        private static int WIDTH = 640;
        private static int HEIGHT = 640;


        public static Color GetColorAt(Point pos, Bitmap bmp)
        {
            return bmp.GetPixel(pos.X, pos.Y);
        }

        public static Bitmap GetBitmap()
        {
            var screenshot = new Bitmap(WIDTH, HEIGHT, PixelFormat.Format24bppRgb);
            using (var graphics = Graphics.FromImage(screenshot))
            {
                var screenWidth = Screen.PrimaryScreen.Bounds.Width;
                var screenHeight = Screen.PrimaryScreen.Bounds.Height;
                var x = (screenWidth - WIDTH) / 2;
                var y = (screenHeight - HEIGHT) / 2;
                graphics.CopyFromScreen(x, y, 0, 0, new Size(WIDTH, HEIGHT), CopyPixelOperation.SourceCopy);
            }
            return screenshot;
        }

        public static Point GetScreenPositionFromBitmapPostion(Point pos)
        {
            return new Point(pos.X + (Screen.PrimaryScreen.Bounds.Width - WIDTH) / 2, pos.Y + (Screen.PrimaryScreen.Bounds.Height - HEIGHT) / 2);
        }

        public static Point GetBitmapPositionFromScreenPosition(Point pos)
        {
            // Check if the screen position is within the canvas bounds
            if (pos.X < (Screen.PrimaryScreen.Bounds.Width - WIDTH) / 2 || pos.X >= (Screen.PrimaryScreen.Bounds.Width + WIDTH) / 2 ||
                pos.Y < (Screen.PrimaryScreen.Bounds.Height - HEIGHT) / 2 || pos.Y >= (Screen.PrimaryScreen.Bounds.Height + HEIGHT) / 2)
            {
                // If the screen position is outside the canvas, return (-1, -1)
                return Point.Empty;
            }
            else
            {
                // Calculate the bitmap position from the screen position
                return new Point(pos.X - (Screen.PrimaryScreen.Bounds.Width - WIDTH) / 2, pos.Y - (Screen.PrimaryScreen.Bounds.Height - HEIGHT) / 2);
            }
        }
    }
}