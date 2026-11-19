
using UnityEngine;

public class HeadRotationController : MonoBehaviour
{
    public float pitch;
    public float yaw;
    public float roll;

    public float rotationSpeed = 5f;

    void Update()
    {
        Quaternion targetRotation =
            Quaternion.Euler(pitch, yaw, roll);

        transform.localRotation = Quaternion.Slerp(
            transform.localRotation,
            targetRotation,
            Mathf.Clamp01(Time.deltaTime * rotationSpeed)
        );
    }

    public void SetHeadRotation(float p, float y, float r)
    {
        pitch = p;
        yaw = y;
        roll = r;
    }
}
