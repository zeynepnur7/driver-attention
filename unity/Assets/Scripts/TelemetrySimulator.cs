
using System;
using UnityEngine;

[Serializable]
public class TelemetryData
{
    public long timestamp;
    public float pitch;
    public float yaw;
    public float roll;
    public float ear;
    public float perclos;
    public bool faceDetected;
    public string alertState;
}

public class TelemetrySimulator : MonoBehaviour
{
    public HeadRotationController headController;
    public AlertIndicatorController alertController;

    [Header("Simulation Settings")]
    public float updateInterval = 2f;
    public bool useSimulation = false;

    private float timer;
    private int sampleIndex;

    private readonly string[] sampleJson =
    {
        @"{
            ""pitch"": 0,
            ""yaw"": 0,
            ""roll"": 0,
            ""ear"": 0.30,
            ""perclos"": 0.02,
            ""faceDetected"": true,
            ""alertState"": ""Normal""
        }",

        @"{
            ""pitch"": 12,
            ""yaw"": 25,
            ""roll"": 5,
            ""ear"": 0.21,
            ""perclos"": 0.15,
            ""faceDetected"": true,
            ""alertState"": ""Caution""
        }",

        @"{
            ""pitch"": 30,
            ""yaw"": -15,
            ""roll"": 10,
            ""ear"": 0.14,
            ""perclos"": 0.45,
            ""faceDetected"": true,
            ""alertState"": ""Critical""
        }"
    };

    void Start()
{
    if (useSimulation)
    {
        ApplyTelemetry(sampleJson[0]);
    }
}

    void Update()
{
    if (!useSimulation)
        return;

    timer += Time.deltaTime;

    if (timer >= updateInterval)
    {
        timer = 0f;

        sampleIndex = (sampleIndex + 1) % sampleJson.Length;

        ApplyTelemetry(sampleJson[sampleIndex]);
    }
}

    public void ApplyTelemetry(string json)
    {
        if (headController == null || alertController == null)
        {
            Debug.LogError("Telemetry references are missing.");
            return;
        }

        try
        {
            TelemetryData data =
                JsonUtility.FromJson<TelemetryData>(json);

            if (data == null)
                return;

            if (data.faceDetected)
            {
                headController.SetHeadRotation(
                    data.pitch,
                    data.yaw,
                    data.roll
                );
            }

            alertController.SetAlertState(data.alertState);

            Debug.Log(
                $"Telemetry: EAR={data.ear}, " +
                $"PERCLOS={data.perclos}, State={data.alertState}"
            );
        }
        catch (Exception e)
        {
            Debug.LogError("Invalid JSON: " + e.Message);
        }
    }
}
