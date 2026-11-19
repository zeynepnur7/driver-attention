
using System.Runtime.InteropServices;
using UnityEngine;

public class WebSocketConnector : MonoBehaviour
{
#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern void ConnectWebSocket();
#endif

    void Start()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        ConnectWebSocket();
#else
        Debug.Log("WebSocket connection will start in WebGL.");
#endif
    }
}
