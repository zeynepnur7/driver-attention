
using UnityEngine;
using UnityEngine.UI;

public class AlertIndicatorController : MonoBehaviour
{
    public enum AlertState
    {
        Normal,
        Caution,
        Critical
    }

    public Image indicatorImage;
    public AlertState currentState = AlertState.Normal;

    void Update()
    {
        if (indicatorImage == null)
            return;

        switch (currentState)
        {
            case AlertState.Normal:
                indicatorImage.color = Color.green;
                break;
            case AlertState.Caution:
                indicatorImage.color = Color.yellow;
                break;
            case AlertState.Critical:
                indicatorImage.color = Color.red;
                break;
        }
    }

    public void SetAlertState(string state)
    {
        if (System.Enum.TryParse(state, true, out AlertState parsed))
        {
            currentState = parsed;
        }
        else
        {
            Debug.LogWarning("Unknown alert state: " + state);
        }
    }
}
