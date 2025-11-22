using System;
using System.Collections;
using System.IO;
using Common;
using UnityEngine;
using UnityEngine.Networking;

public class DataConnector : IDataReceiver
{
    [SerializeField] private string URI = "http://127.0.0.1:8000/pointcloud";
    [SerializeField] private string imageFileName = "sample.png"; // 送信する画像ファイル名（StreamingAssets 等に配置）

    // IDataReceiver 実装: 外部からは callback を渡して呼び出す
    IEnumerator IDataReceiver.GetData(Action<PicturePoints> callback)
    {
        string path = ResolveImagePath(imageFileName);
        return FetchData(path, callback);
    }

    // 画像 → サーバ送信 → byte(float32*5N) 受信 → PicturePoints
    public IEnumerator FetchData(string imagePath, Action<PicturePoints> callback)
    {
        if (string.IsNullOrEmpty(imagePath))
        {
            callback?.Invoke(EmptyPoints());
            yield break;
        }
        if (!File.Exists(imagePath))
        {
            Debug.LogWarning("Image not found: " + imagePath);
            callback?.Invoke(EmptyPoints());
            yield break;
        }

        byte[] fileBytes;
        try
        {
            fileBytes = File.ReadAllBytes(imagePath);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Read error: " + e.Message);
            callback?.Invoke(EmptyPoints());
            yield break;
        }

        var form = new WWWForm();
        form.AddBinaryData("file", fileBytes, Path.GetFileName(imagePath));

        using (var req = UnityWebRequest.Post(URI, form))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

#if UNITY_2020_2_OR_NEWER
            if (req.result != UnityWebRequest.Result.Success)
#else
            if (req.isHttpError || req.isNetworkError)
#endif
            {
                Debug.LogWarning("Request failed: " + req.error);
                callback?.Invoke(EmptyPoints());
                yield break;
            }

            byte[] data = req.downloadHandler.data;
            if (data == null || data.Length == 0)
            {
                Debug.LogWarning("Empty response.");
                callback?.Invoke(EmptyPoints());
                yield break;
            }

            PicturePoints pts;
            try
            {
                pts = ParsePointCloud(data);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Parse error: " + e.Message);
                pts = EmptyPoints();
            }

            callback?.Invoke(pts);
        }
    }

    private static PicturePoints ParsePointCloud(byte[] bytes)
    {
        int floatCount = bytes.Length / 4;
        if (floatCount == 0 || floatCount % 5 != 0)
            throw new Exception("Invalid float count: " + floatCount);

        float[] floats = new float[floatCount];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);

        int pointCount = floatCount / 5;
        var points = new Point[pointCount];

        for (int i = 0; i < pointCount; i++)
        {
            int idx = i * 5;
            // 既に正規化されている前提。念のため軽い clamp。
            float x = Clamp01(floats[idx]);
            float y = Clamp01(floats[idx + 1]);
            float r = Clamp01(floats[idx + 2]);
            float g = Clamp01(floats[idx + 3]);
            float b = Clamp01(floats[idx + 4]);

            points[i] = new Point
            {
                pos = new Vector2(x, y),
                color = new Color(r, g, b, 1f)
            };
        }

        // 解像度は暫定値
        return new PicturePoints(points, new Vector2Int(1, 1));
    }

    private static PicturePoints EmptyPoints() =>
        new PicturePoints(Array.Empty<Point>(), new Vector2Int(0, 0));

    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);

    private static string ResolveImagePath(string name)
    {
        if (string.IsNullOrEmpty(name)) return null;
        if (Path.IsPathRooted(name) && File.Exists(name)) return name;
        if (File.Exists(name)) return Path.GetFullPath(name);

        string sa = Path.Combine(Application.streamingAssetsPath, name);
        if (File.Exists(sa)) return sa;

        string pd = Path.Combine(Application.persistentDataPath, name);
        if (File.Exists(pd)) return pd;

#if UNITY_EDITOR
        string assets = Path.Combine(Application.dataPath, name);
        if (File.Exists(assets)) return assets;
#endif
        return null;
    }
}
