export async function GET() {
  return Response.json(
    {
      error: "Trainer stream is not implemented yet."
    },
    {
      status: 501
    }
  );
}
